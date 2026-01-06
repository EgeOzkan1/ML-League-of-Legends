"""
Phase 3 (Final Defense-ready): League of Legends match outcome prediction at 10 minutes
Dataset: high_diamond_ranked_10min.csv
Target:  blueWins (1 = Blue team wins, 0 = Red team wins)

Deliverables (rubric-aligned):
- Compare 2 algorithms: KNN vs Random Forest
- Evaluation beyond accuracy: Confusion Matrix + F1-score (required) + ROC-AUC (added)
- Interpretation: Random Forest feature importance (kept) + permutation importance (added, model-agnostic)
- Game/business insight: actionable interpretation with non-causal wording

Important note:
- Feature importance indicates association/predictive power, not causation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("=" * 80)
print("PHASE 3: FINAL DEFENSE - MODEL COMPARISON AND EVALUATION")
print("=" * 80)
print("\nLoading data...")

df = pd.read_csv("high_diamond_ranked_10min.csv")

# Separate features and target
# Exclude gameId as it's just an identifier and shouldn't affect predictions
X = df.drop(["blueWins", "gameId"], axis=1)
y = df["blueWins"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling:
# - KNN is distance-based, so scaling prevents large-scale features from dominating distances.
# - Random Forest is tree-based, so scaling is not required.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# ============================================================================
# MODEL 1: K-NEAREST NEIGHBORS (KNN)
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 1: K-NEAREST NEIGHBORS (KNN)")
print("=" * 80)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predictions
y_pred_knn = knn.predict(X_test_scaled)
y_pred_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
roc_auc_knn = roc_auc_score(y_test, y_pred_proba_knn)

print(f"\nKNN Results:")
print(f"  Accuracy:  {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")
print(f"  F1-Score:  {f1_knn:.4f}")
print(f"  Precision: {precision_knn:.4f}")
print(f"  Recall:    {recall_knn:.4f}")
print(f"  ROC-AUC:   {roc_auc_knn:.4f}")

# ============================================================================
# MODEL 2: RANDOM FOREST
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 2: RANDOM FOREST")
print("=" * 80)

# Train Random Forest model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# Evaluation metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"\nRandom Forest Results:")
print(f"  Accuracy:  {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
print(f"  F1-Score:  {f1_rf:.4f}")
print(f"  Precision: {precision_rf:.4f}")
print(f"  Recall:    {recall_rf:.4f}")
print(f"  ROC-AUC:   {roc_auc_rf:.4f}")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC'],
    'KNN': [accuracy_knn, f1_knn, precision_knn, recall_knn, roc_auc_knn],
    'Random Forest': [accuracy_rf, f1_rf, precision_rf, recall_rf, roc_auc_rf]
})

comparison_df['Difference'] = comparison_df['Random Forest'] - comparison_df['KNN']
comparison_df['Winner'] = comparison_df.apply(
    lambda row: 'Random Forest' if row['Difference'] > 0 else 'KNN', 
    axis=1
)

print("\nDetailed Comparison:")
print(comparison_df.to_string(index=False))

# Determine overall winner (defensible decision rule for defense):
# Primary metric = F1-score (balances precision/recall and is robust to class imbalance).
# Secondary tie-breaker = Accuracy.
decision_rule = "Highest F1-score wins (Accuracy used only as tie-breaker)"

if f1_rf > f1_knn:
    winner = "Random Forest"
elif f1_knn > f1_rf:
    winner = "KNN"
else:
    winner = "Random Forest" if accuracy_rf >= accuracy_knn else "KNN"

if winner == "Random Forest":
    winner_metrics = {
        'accuracy': accuracy_rf,
        'f1': f1_rf,
        'precision': precision_rf,
        'recall': recall_rf,
        'roc_auc': roc_auc_rf
    }
    loser = "KNN"
    loser_metrics = {
        'accuracy': accuracy_knn,
        'f1': f1_knn,
        'precision': precision_knn,
        'recall': recall_knn,
        'roc_auc': roc_auc_knn
    }
else:
    winner_metrics = {
        'accuracy': accuracy_knn,
        'f1': f1_knn,
        'precision': precision_knn,
        'recall': recall_knn,
        'roc_auc': roc_auc_knn
    }
    loser = "Random Forest"
    loser_metrics = {
        'accuracy': accuracy_rf,
        'f1': f1_rf,
        'precision': precision_rf,
        'recall': recall_rf,
        'roc_auc': roc_auc_rf
    }

print(f"\n{'='*80}")
print(f"OVERALL WINNER: {winner.upper()}")
print(f"{'='*80}")
print(f"\nDecision rule: {decision_rule}")
print(f"\n{winner} is selected over {loser} primarily due to higher F1-score:")
print(f"  - Accuracy:  {winner_metrics['accuracy']:.4f} vs {loser_metrics['accuracy']:.4f}")
print(f"  - F1-Score: {winner_metrics['f1']:.4f} vs {loser_metrics['f1']:.4f}")
print(f"  - ROC-AUC:  {winner_metrics['roc_auc']:.4f} vs {loser_metrics['roc_auc']:.4f}")

# ============================================================================
# VISUALIZATION: CONFUSION MATRICES
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# KNN Confusion Matrix
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Red Wins', 'Blue Wins'],
            yticklabels=['Red Wins', 'Blue Wins'])
axes[0].set_title(f'KNN Confusion Matrix\nAccuracy: {accuracy_knn:.4f}, F1: {f1_knn:.4f}', 
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=11)
axes[0].set_ylabel('Actual', fontsize=11)

# Random Forest Confusion Matrix
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Red Wins', 'Blue Wins'],
            yticklabels=['Red Wins', 'Blue Wins'])
axes[1].set_title(f'Random Forest Confusion Matrix\nAccuracy: {accuracy_rf:.4f}, F1: {f1_rf:.4f}', 
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=11)
axes[1].set_ylabel('Actual', fontsize=11)

plt.tight_layout()
plt.savefig('Phase3_Confusion_Matrices.png', dpi=300, bbox_inches='tight')
print("Saved: Phase3_Confusion_Matrices.png")
plt.close()

# ============================================================================
# VISUALIZATION: ROC CURVES (uses predict_proba outputs)
# ============================================================================

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_proba_knn)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

plt.figure(figsize=(10, 7))
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.3f})', linewidth=2, color='#3498db')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})', linewidth=2, color='#2ecc71')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Random baseline')
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves: KNN vs Random Forest', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Phase3_ROC_Curves.png', dpi=300, bbox_inches='tight')
print("Saved: Phase3_ROC_Curves.png")
plt.close()

# ============================================================================
# VISUALIZATION: METRIC COMPARISON
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(comparison_df))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_df['KNN'], width, label='KNN', alpha=0.8, color='#3498db')
bars2 = ax.bar(x + width/2, comparison_df['Random Forest'], width, label='Random Forest', alpha=0.8, color='#2ecc71')

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison: KNN vs Random Forest', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Metric'])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('Phase3_Metric_Comparison.png', dpi=300, bbox_inches='tight')
print("Saved: Phase3_Metric_Comparison.png")
plt.close()

# ============================================================================
# FEATURE IMPORTANCE (Random Forest)
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Get feature importances from Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Visualize feature importance
fig, ax = plt.subplots(figsize=(12, 10))

top_features = feature_importance.head(15)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'], fontsize=10)
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax.text(row['Importance'], i, f' {row["Importance"]:.4f}',
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('Phase3_Feature_Importance.png', dpi=300, bbox_inches='tight')
print("Saved: Phase3_Feature_Importance.png")
plt.close()

# ============================================================================
# MODEL-AGNOSTIC FEATURE IMPORTANCE: PERMUTATION IMPORTANCE (WINNER MODEL)
# ============================================================================

print("\n" + "=" * 80)
print("PERMUTATION IMPORTANCE (MODEL-AGNOSTIC INTERPRETATION)")
print("=" * 80)
print("Note: permutation importance measures predictive association, not causation.")

feature_names = X.columns.tolist()

if winner == "Random Forest":
    perm_model = rf
    X_perm = X_test.copy()  # unscaled for RF
else:
    perm_model = knn
    # Use scaled features for KNN (distance-based model). Keep column labels for readable outputs.
    X_perm = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

perm = permutation_importance(
    perm_model,
    X_perm,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance_Mean": perm.importances_mean,
    "Importance_Std": perm.importances_std
}).sort_values("Importance_Mean", ascending=False)

perm_importance.to_csv('Phase3_Permutation_Importance.csv', index=False)
print("Saved: Phase3_Permutation_Importance.csv")

top15_perm = perm_importance.head(15).iloc[::-1]
plt.figure(figsize=(12, 10))
plt.barh(
    top15_perm["Feature"],
    top15_perm["Importance_Mean"],
    xerr=top15_perm["Importance_Std"],
    color=plt.cm.plasma(np.linspace(0.1, 0.9, len(top15_perm))),
    alpha=0.9
)
plt.xlabel("Permutation Importance (mean ΔF1)", fontsize=12, fontweight='bold')
plt.title(f"Top 15 Permutation Importances (Winning Model: {winner})", fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('Phase3_Permutation_Importance.png', dpi=300, bbox_inches='tight')
print("Saved: Phase3_Permutation_Importance.png")
plt.close()

# ============================================================================
# BUSINESS/GAME INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("BUSINESS/GAME INSIGHTS")
print("=" * 80)

print("\nImportant note: Feature importance indicates association/predictive power, not causation.")

print("\n1. KEY PREDICTORS OF MATCH OUTCOME:")
print("-" * 80)
top_5_features = feature_importance.head(5)
for idx, row in top_5_features.iterrows():
    print(f"   - {row['Feature']}: {row['Importance']:.4f}")

print("\n2. INTERPRETATION:")
print("-" * 80)

# Analyze top features
top_feature = feature_importance.iloc[0]['Feature']
top_importance = feature_importance.iloc[0]['Importance']

print(f"   The most important feature is '{top_feature}' with importance {top_importance:.4f}.")
print(f"   This indicates that '{top_feature}' is one of the strongest predictive signals for match outcome at 10 minutes.")

# Check if gold/experience differences are important
gold_features = feature_importance[feature_importance['Feature'].str.contains('Gold', case=False)]
exp_features = feature_importance[feature_importance['Feature'].str.contains('Experience', case=False)]

print(f"\n3. ECONOMIC FACTORS:")
print("-" * 80)
if len(gold_features) > 0:
    print(f"   Gold-related features are highly predictive:")
    for idx, row in gold_features.head(3).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")

if len(exp_features) > 0:
    print(f"\n   Experience-related features:")
    for idx, row in exp_features.head(3).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")

# Check objective control
objective_features = feature_importance[
    feature_importance['Feature'].str.contains('Dragon|Herald|Tower', case=False)
]

print(f"\n4. OBJECTIVE CONTROL:")
print("-" * 80)
if len(objective_features) > 0:
    print(f"   Objective control features:")
    for idx, row in objective_features.head(5).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")
else:
    print("   Objective features have lower importance compared to economic factors.")

print("\n5. RECOMMENDATIONS FOR GAME BALANCE:")
print("-" * 80)
print("   Based on the model analysis:")
print("   - Focus on balancing economic advantages (gold/experience differences)")
print("   - Early-game gold leads are strong predictive signals of match outcome")
print("   - Consider adjusting early-game objective rewards if they create snowball effects")
print("   - The most predictive signals at 10 minutes appear to be related to early economy and tempo")
print("   - Teams aiming to maximize win probability should prioritize consistent early gold/XP generation and low-risk objective setups")

print("\n6. MODEL PERFORMANCE INSIGHTS:")
print("-" * 80)
print(f"   - {winner} achieved {winner_metrics['accuracy']*100:.2f}% accuracy")
print(f"   - F1-Score of {winner_metrics['f1']:.4f} indicates good balance between precision and recall")
print(f"   - The model can predict match outcomes with reasonable confidence at 10 minutes")
print(f"   - This indicates early-game signals (at 10 minutes) are strongly associated with final outcomes in this dataset")

# ============================================================================
# DETAILED CLASSIFICATION REPORTS
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 80)

print("\nKNN Classification Report:")
print("-" * 80)
print(classification_report(y_test, y_pred_knn, 
                            target_names=['Red Wins', 'Blue Wins']))

print("\nRandom Forest Classification Report:")
print("-" * 80)
print(classification_report(y_test, y_pred_rf,
                            target_names=['Red Wins', 'Blue Wins']))

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS...")
print("=" * 80)

# Save comparison results
comparison_df.to_csv('Phase3_Model_Comparison.csv', index=False)
print("Saved: Phase3_Model_Comparison.csv")

# Save feature importance
feature_importance.to_csv('Phase3_Feature_Importance.csv', index=False)
print("Saved: Phase3_Feature_Importance.csv")

# Save detailed results
results_summary = {
    'Model': ['KNN', 'Random Forest', f'Winner ({winner} by F1)'],
    'Accuracy': [accuracy_knn, accuracy_rf, winner_metrics['accuracy']],
    'F1_Score': [f1_knn, f1_rf, winner_metrics['f1']],
    'Precision': [precision_knn, precision_rf, winner_metrics['precision']],
    'Recall': [recall_knn, recall_rf, winner_metrics['recall']],
    'ROC_AUC': [roc_auc_knn, roc_auc_rf, winner_metrics['roc_auc']],
    'Decision_Rule': ['', '', decision_rule]
}
results_df = pd.DataFrame(results_summary)
results_df.to_csv('Phase3_Results_Summary.csv', index=False)
print("Saved: Phase3_Results_Summary.csv")

print("\n" + "=" * 80)
print("PHASE 3 COMPLETE!")
print("=" * 80)
print(f"\nWinner: {winner}")
print(f"Decision rule: {decision_rule}")
print(f"Best Accuracy: {winner_metrics['accuracy']*100:.2f}%")
print(f"Best F1-Score: {winner_metrics['f1']:.4f}")
print(f"Best ROC-AUC:  {winner_metrics['roc_auc']:.4f}")
print("\nAll visualizations and results have been saved.")

