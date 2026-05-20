#!/usr/bin/env python3
"""
Project-016 GNN Drug Repositioning - Top-20 Prediction & Literature Validation
加载训练好的模型，预测所有未见药物-疾病对的TMJOA治疗潜力
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from collections import defaultdict
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 64
HIDDEN_DIM = 128


# ========== Model (same as training) ==========

class DrugRepositionGNN(torch.nn.Module):
    def __init__(self, hidden_channels=HIDDEN_DIM, out_channels=EMB_DIM, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i, _ in enumerate(range(num_layers)):
            in_dim = EMB_DIM if i == 0 else -1  # First layer 64, rest auto
            conv = HeteroConv({
                ('drug', 'treats', 'disease'): SAGEConv((EMB_DIM if i==0 else HIDDEN_DIM, EMB_DIM if i==0 else HIDDEN_DIM), hidden_channels),
                ('disease', 'treated_by', 'drug'): SAGEConv((EMB_DIM if i==0 else HIDDEN_DIM, EMB_DIM if i==0 else HIDDEN_DIM), hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        return x_dict

    def predict_edge(self, drug_emb, disease_emb):
        x = torch.cat([drug_emb, disease_emb], dim=-1)
        return self.predictor(x).squeeze(-1)


def load_model_and_data():
    """加载训练好的模型和训练集数据"""
    # 加载检查点
    checkpoint = torch.load('.tmp/p016_gnn_best.pt', map_location=DEVICE, weights_only=False)
    
    drug2idx = checkpoint['drug2idx']
    disease2idx = checkpoint['disease2idx']
    
    # 加载v3.9d训练集（获取已知边）
    with open('.tmp/p016_train_v4_2.json') as f:
        data = json.load(f)
    
    all_samples = data['splits']['train'] + data['splits']['val'] + data['splits']['test']
    
    # 构建图
    num_drugs = len(drug2idx)
    num_diseases = len(disease2idx)
    
    graph = HeteroData()
    graph['drug'].x = torch.randn(num_drugs, EMB_DIM) * 0.1
    graph['disease'].x = torch.randn(num_diseases, EMB_DIM) * 0.1
    
    # 所有已知边（用于消息传递）
    drug_idx = [drug2idx[s['drug']] for s in all_samples]
    disease_idx = [disease2idx[s['disease']] for s in all_samples]
    
    edge_index = torch.tensor([drug_idx, disease_idx], dtype=torch.long)
    graph['drug', 'treats', 'disease'].edge_index = edge_index
    graph['disease', 'treated_by', 'drug'].edge_index = edge_index.flip(0)
    
    graph = graph.to(DEVICE)
    
    # 初始化并加载模型
    model = DrugRepositionGNN().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 已知边集合（排除已知的）
    known_edges = set()
    for s in all_samples:
        known_edges.add((s['drug'], s['disease']))
    
    return model, graph, drug2idx, disease2idx, known_edges, all_samples


def predict_all_pairs(model, graph, drug2idx, disease2idx, known_edges):
    """对所有未见药物-疾病对进行预测"""
    idx2drug = {v: k for k, v in drug2idx.items()}
    idx2disease = {v: k for k, v in disease2idx.items()}
    
    with torch.no_grad():
        x_dict = model(graph.x_dict, graph.edge_index_dict)
        drug_emb = x_dict['drug']
        disease_emb = x_dict['disease']
    
    predictions = []
    
    for drug_name, drug_i in drug2idx.items():
        for disease_name, disease_i in disease2idx.items():
            if (drug_name, disease_name) in known_edges:
                continue
            
            d_emb = drug_emb[drug_i].unsqueeze(0)
            dis_emb = disease_emb[disease_i].unsqueeze(0)
            
            with torch.no_grad():
                score = torch.sigmoid(model.predict_edge(d_emb, dis_emb)).item()
            
            predictions.append({
                'drug': drug_name,
                'disease': disease_name,
                'score': score,
                'is_tmj': 'tmj' in disease_name.lower() or 'tmd' in disease_name.lower() or 'temporomandibular' in disease_name.lower(),
            })
    
    return predictions


def analyze_top_predictions(predictions, all_samples, top_k=20):
    """分析Top-K预测结果"""
    # 按分数排序
    predictions_sorted = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # 药物统计
    drug_freq = defaultdict(int)
    for p in predictions_sorted[:top_k*3]:
        drug_freq[p['drug']] += 1
    
    # TMJ相关Top预测
    tmj_predictions = [p for p in predictions_sorted if p['is_tmj']]
    
    # 已有训练集中的TMJ正样本
    tmj_positive = [s for s in all_samples if s['label'] > 0 and 
                    ('tmj' in s['disease'].lower() or 'tmd' in s['disease'].lower())]
    
    print("=" * 70)
    print("Project-016 GNN Drug Repositioning - Top-20 Predictions")
    print("=" * 70)
    
    # 全局Top-20
    print(f"\n{'='*70}")
    print(f"Top-{top_k} 全局预测（所有疾病）")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Drug':<30} {'Disease':<25} {'Score':<8} {'Note'}")
    print("-" * 70)
    
    for i, p in enumerate(predictions_sorted[:top_k], 1):
        note = "⭐ NEW" if not p['is_tmj'] else "🦷 TMJ"
        print(f"{i:<6} {p['drug']:<30} {p['disease']:<25} {p['score']:.4f}  {note}")
    
    # TMJ-相关Top-20
    print(f"\n{'='*70}")
    print(f"Top-{top_k} TMJ/TMD/TMJOA 相关预测")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Drug':<30} {'Disease':<25} {'Score':<8} {'Status'}")
    print("-" * 70)
    
    for i, p in enumerate(tmj_predictions[:top_k], 1):
        # 检查是否在已知正样本中
        in_train = any(s['drug'] == p['drug'] and s['disease'] == p['disease'] 
                        for s in tmj_positive)
        status = "✅ Known" if in_train else "🔮 Novel"
        print(f"{i:<6} {p['drug']:<30} {p['disease']:<25} {p['score']:.4f}  {status}")
    
    # 统计
    print(f"\n{'='*70}")
    print("统计摘要")
    print(f"{'='*70}")
    print(f"总预测对数: {len(predictions)}")
    print(f"TMJ相关预测: {len(tmj_predictions)}")
    print(f"Top-20平均分数: {np.mean([p['score'] for p in predictions_sorted[:top_k]]):.4f}")
    print(f"TMJ Top-20平均分数: {np.mean([p['score'] for p in tmj_predictions[:top_k]]):.4f}")
    print(f"\n高频候选药物（Top-60中）:")
    for drug, freq in sorted(drug_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {drug}: {freq}次")
    
    return predictions_sorted, tmj_predictions


def save_report(predictions_sorted, tmj_predictions, all_samples):
    """保存预测报告"""
    report = {
        'metadata': {
            'model': 'DrugRepositionGNN (HeteroSAGE)',
            'checkpoint': '.tmp/p016_gnn_best.pt',
            'best_val_auc': 0.9714,
            'total_predictions': len(predictions_sorted),
            'date': '2026-05-18',
        },
        'top20_global': predictions_sorted[:20],
        'top20_tmj': tmj_predictions[:20],
        'tmj_positive_in_train': [
            {'drug': s['drug'], 'disease': s['disease'], 'label': s['label']}
            for s in all_samples
            if s['label'] > 0 and ('tmj' in s['disease'].lower() or 'tmd' in s['disease'].lower())
        ][:30],
    }
    
    with open('.tmp/p016_gnn_top20_predictions.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 报告已保存: .tmp/p016_gnn_top20_predictions.json")


def main():
    print("=" * 70)
    print("Project-016 GNN Drug Repositioning - Prediction Pipeline")
    print("=" * 70)
    
    # 加载
    print("\n[1] 加载模型和数据...")
    model, graph, drug2idx, disease2idx, known_edges, all_samples = load_model_and_data()
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  药物数: {len(drug2idx)}, 疾病数: {len(disease2idx)}")
    print(f"  已知边: {len(known_edges)}")
    
    # 预测
    print("\n[2] 预测所有未见药物-疾病对...")
    predictions = predict_all_pairs(model, graph, drug2idx, disease2idx, known_edges)
    print(f"  预测对数: {len(predictions)}")
    
    # 分析
    print("\n[3] 分析Top-20预测...")
    predictions_sorted, tmj_predictions = analyze_top_predictions(predictions, all_samples)
    
    # 保存
    print("\n[4] 保存报告...")
    save_report(predictions_sorted, tmj_predictions, all_samples)
    
    print(f"\n{'='*70}")
    print("预测完成。下一步：对Top候选进行文献回溯验证")
    print(f"{'='*70}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
