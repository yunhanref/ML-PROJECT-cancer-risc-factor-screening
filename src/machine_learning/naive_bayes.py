import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score

def main():
    print("Bayesian Approach (Naive Bayes) Model Training starting...")
    
    # 1. Load the preprocessed datasets
    X_train = pd.read_csv('X_train_hazir.csv')
    X_test  = pd.read_csv('X_test_hazir.csv')
    y_train = pd.read_csv('y_train_hazir.csv').squeeze().astype(int)
    y_test  = pd.read_csv('y_test_hazir.csv').squeeze().astype(int)
    
    print(f"Data shapes: Train={X_train.shape}, Test={X_test.shape}")
    
    
    model_nb = GaussianNB()
    
    param_grid = {
        'var_smoothing': np.logspace(-11, -1, 100)
    }
    
    grid = GridSearchCV(
        model_nb,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("Grid Search for Naive Bayes starts...")
    grid.fit(X_train, y_train)
    
    best_nb = grid.best_estimator_
    print(f"Best var_smoothing: {grid.best_params_['var_smoothing']:.2e}")
    print(f"Best CV F1-score: {grid.best_score_:.3f}")
    
    
    y_pred = best_nb.predict(X_test)
    y_prob = best_nb.predict_proba(X_test)[:, 1]
    
    
    print("\nModel Performance Metrics:")
    print(classification_report(y_test, y_pred))
    
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"Test F1-score : {test_f1:.3f}")
    print(f"Test ROC-AUC  : {test_auc:.3f}")
    
   
    os.makedirs('models', exist_ok=True)
    model_path = 'models/model_nb.pkl'
    joblib.dump(best_nb, model_path)
    print(f"\nSaved model: {model_path}")
    
    
    np.save('y_pred_nb.npy', y_pred)
    np.save('y_prob_nb.npy', y_prob)
    print("Saved predictions: y_pred_nb.npy, y_prob_nb.npy")
    
    
    analysis_dir = 'analysis/naive_bayes'
    os.makedirs(analysis_dir, exist_ok=True)
    
   
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Kanser Yok', 'Kanser Var'], 
        yticklabels=['Kanser Yok', 'Kanser Var'],
        cbar=False,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title('Naive Bayes - Konfüzyon Matrisi', fontsize=14, pad=15)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.tight_layout()
    cm_plot_path = os.path.join(analysis_dir, 'konfüzyon.png')
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix plot: {cm_plot_path}")
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (Eğri Altı Alan = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)', fontsize=11)
    plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)', fontsize=11)
    plt.title('Naive Bayes - ROC Eğrisi', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(analysis_dir, 'roc.png')
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curve plot: {roc_plot_path}")
    
    print("\nBayesian Approach model training and analysis successfully completed!")

if __name__ == '__main__':
    main()
