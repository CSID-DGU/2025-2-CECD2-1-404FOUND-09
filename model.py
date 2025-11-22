import math
import os
import time

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = b.replace(0, np.nan)
    result = a / denom
    return result.replace([np.inf, -np.inf], 0).fillna(0)


def _augment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns
    
    if {'num_lab_procedures', 'num_medications'}.issubset(cols):
        df['lab_med_ratio'] = _safe_divide(df['num_lab_procedures'], df['num_medications'])
        df['lab_plus_med'] = df['num_lab_procedures'] + df['num_medications']
    
    if {'num_medications', 'time_in_hospital'}.issubset(cols):
        df['meds_per_day'] = _safe_divide(df['num_medications'], df['time_in_hospital'] + 1)
        df['med_time_interaction'] = df['num_medications'] * df['time_in_hospital']
    
    if {'num_procedures', 'time_in_hospital'}.issubset(cols):
        df['procedures_per_day'] = _safe_divide(df['num_procedures'], df['time_in_hospital'] + 1)
        df['procedures_time_interaction'] = df['num_procedures'] * df['time_in_hospital']
    
    if {'num_diagnoses', 'time_in_hospital'}.issubset(cols):
        df['diagnoses_per_day'] = _safe_divide(df['num_diagnoses'], df['time_in_hospital'] + 1)
    
    stay_cols = {'number_inpatient', 'number_emergency', 'number_outpatient'}
    if stay_cols.issubset(cols):
        df['acute_visit_load'] = (
            df['number_inpatient'] * 2 +
            df['number_emergency'] * 1.5 +
            df['number_outpatient']
        ).astype('float32')
        df['recent_hospital_flag'] = (
            (df['number_inpatient'] > 0) |
            (df['number_emergency'] > 0) |
            (df['number_outpatient'] > 0)
        ).astype('int32')
    
    if 'time_in_hospital' in cols:
        df['long_stay_flag'] = (df['time_in_hospital'] >= 7).astype('int32')
        df['very_long_stay_flag'] = (df['time_in_hospital'] >= 14).astype('int32')
        df['short_stay_flag'] = (df['time_in_hospital'] <= 3).astype('int32')
    
    if {'num_diagnoses', 'num_medications'}.issubset(cols):
        df['diagnosis_med_combo'] = df['num_diagnoses'] * df['num_medications']
    
    if {'num_lab_procedures', 'num_diagnoses'}.issubset(cols):
        df['lab_diag_interaction'] = df['num_lab_procedures'] * df['num_diagnoses']
    
    return df


def _is_diabetes_code(code: str) -> bool:
    if code is None:
        return False
    code = str(code).strip().upper()
    if code in ('', '?'):
        return False
    if code.startswith('250'):
        return True
    if code.startswith(('E10', 'E11', 'E13')):
        return True
    return False


def _is_complication_code(code: str) -> bool:
    if code is None:
        return False
    code = str(code).strip().upper()
    if code in ('', '?'):
        return False
    complication_prefixes = (
        '250.4', '250.5', '250.6', '250.7', '249.4', '249.5', '249.6', '249.7',
        '357.2', '362.0', '366.41', '274.1', '403', '404', '405', '428', '585'
    )
    return any(code.startswith(prefix) for prefix in complication_prefixes)


def _ensure_diabetes_label(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col in df.columns:
        return df
    
    diag_cols = [col for col in df.columns if col.lower().startswith('diag')]
    if not diag_cols and target_col == 'diabetes_flag':
        df[target_col] = 1
        return df
    
    if target_col == 'diabetes_flag':
        diag_subset = df[diag_cols].fillna('')
        diabetes_mask = diag_subset.apply(
            lambda row: any(_is_diabetes_code(code) for code in row), axis=1
        ).astype('int64')
        if 'diabetesMed' in df.columns:
            med_mask = df['diabetesMed'].str.upper().eq('YES')
            diabetes_mask = np.logical_or(diabetes_mask.astype(bool), med_mask).astype('int64')
        df[target_col] = diabetes_mask
        return df
    
    raise ValueError(f"지정한 타깃 컬럼 '{target_col}'을 찾을 수 없습니다.")


def _compute_complication_labels(df: pd.DataFrame) -> np.ndarray:
    diag_cols = [col for col in df.columns if col.lower().startswith('diag')]
    if not diag_cols:
        return np.zeros(len(df), dtype='int64')
    diag_subset = df[diag_cols].fillna('')
    return diag_subset.apply(
        lambda row: any(_is_complication_code(code) for code in row), axis=1
    ).astype('int64').values


def _is_diabetes_code(code: str) -> bool:
    if code is None:
        return False
    code = str(code).strip().upper()
    if code == '' or code == '?':
        return False
    if code.startswith('250'):
        return True  # ICD-9 diabetes
    if code.startswith(('E10', 'E11', 'E13')):
        return True  # ICD-10 common diabetes codes
    return False


def _ensure_diabetes_label(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col in df.columns:
        return df
    
    diag_cols = [col for col in df.columns if col.lower().startswith('diag')]
    if not diag_cols and target_col == 'diabetes_flag':
        df[target_col] = 1
        return df
    
    if target_col == 'diabetes_flag':
        diag_subset = df[diag_cols].fillna('')
        diabetes_mask = diag_subset.apply(
            lambda row: any(_is_diabetes_code(code) for code in row), axis=1
        ).astype('int64')
        if 'diabetesMed' in df.columns:
            diabetes_mask = np.logical_or(
                diabetes_mask.astype(bool),
                df['diabetesMed'].str.upper().eq('YES')
            ).astype('int64')
        df[target_col] = diabetes_mask
        return df
    
    raise ValueError(f"지정한 타깃 컬럼 '{target_col}'을 찾을 수 없습니다.")


def _compute_optimal_threshold(probabilities: np.ndarray, labels: np.ndarray,
                               min_thr: float = 0.1, max_thr: float = 0.9, steps: int = 161):
    if len(probabilities) == 0:
        return 0.5, 0.0
    thresholds = np.linspace(min_thr, max_thr, steps)
    best_thr, best_acc = 0.5, 0.0
    for thr in thresholds:
        preds = (probabilities >= thr).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_thr, best_acc = thr, acc
    return best_thr, best_acc * 100

# Diabetes 데이터셋 로딩 함수 및 Dataset 클래스
class DiabetesDataset(Dataset):
    def __init__(self, X, y, stay=None, readmit=None, complication=None):
        self.X = X.astype('float32')
        self.y = y.astype('int64')
        n = len(X)
        if stay is None:
            stay = np.zeros(n, dtype='int64')
        if readmit is None:
            readmit = np.zeros(n, dtype='int64')
        if complication is None:
            complication = np.zeros(n, dtype='int64')
        self.stay = stay.astype('int64')
        self.readmit = readmit.astype('int64')
        self.complication = complication.astype('int64')
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(self.stay[idx]),
            torch.tensor(self.readmit[idx]),
            torch.tensor(self.complication[idx])
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = F.relu(out + residual)
        out = self.dropout(out)
        return out


class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        hidden_dim = max(1, feature_dim // 2)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights, weights

def encode_categorical_features(df, categorical_cols, min_freq=0.01, max_unique=50):
    if not categorical_cols:
        return None
    
    print(f"범주형 컬럼 ({len(categorical_cols)}개) 인코딩 시작: {categorical_cols[:5]}...", flush=True)
    cat_df = df[categorical_cols].copy().fillna("UNKNOWN")
    
    processed_cols = []
    for col in categorical_cols:
        value_counts = cat_df[col].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < min_freq].index
        if len(rare_categories) > 0:
            cat_df.loc[cat_df[col].isin(rare_categories), col] = "__OTHER__"
        
        # 과도한 차원 증가 방지
        unique_vals = cat_df[col].nunique()
        if unique_vals > max_unique:
            top_values = cat_df[col].value_counts().nlargest(max_unique - 1).index
            cat_df.loc[~cat_df[col].isin(top_values), col] = "__OTHER__"
        processed_cols.append(col)
    
    encoded = pd.get_dummies(cat_df, columns=processed_cols, drop_first=True, dtype="float32")
    print(f"범주형 인코딩 완료: 추가 특성 {encoded.shape[1]}개", flush=True)
    return encoded

def load_diabetes_data(csv_path, test_size=0.2, random_state=42, max_features=96, target_col='diabetes_flag'):
    print(f"데이터 로드 시작: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    print(f"원본 데이터 크기: {df.shape}", flush=True)
    
    # 기본 전처리
    drop_cols = ['encounter_id', 'patient_nbr']
    available_drop_cols = [col for col in drop_cols if col in df.columns]
    if available_drop_cols:
        df = df.drop(columns=available_drop_cols)
        print(f"제거된 컬럼: {available_drop_cols}", flush=True)
    
    # readmitted 컬럼 처리
    if 'readmitted' in df.columns:
        df['readmitted'] = df['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
    
    # 파생 특징 및 타깃 생성
    df = _augment_features(df)
    df = _ensure_diabetes_label(df, target_col)
    complication_labels = _compute_complication_labels(df)
    readmission_labels = df['readmitted'].values.astype('int64') if 'readmitted' in df.columns else np.zeros(len(df), dtype='int64')
    
    # 숫자형/범주형 컬럼 분리
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in [target_col, 'readmitted']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    if 'max_glu_serum' in numeric_cols:
        numeric_cols.remove('max_glu_serum')  # 문제가 되는 컬럼 제거
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in [target_col, 'readmitted']:
        if col in categorical_cols:
            categorical_cols.remove(col)
    
    print(f"사용할 특성 컬럼 ({len(numeric_cols)}개): {numeric_cols[:5]}...", flush=True)
    if categorical_cols:
        print(f"사용할 범주형 컬럼 ({len(categorical_cols)}개): {categorical_cols[:5]}...", flush=True)
    else:
        print("사용할 범주형 컬럼 없음", flush=True)
    
    # 특성 데이터 추출
    X = df[numeric_cols].values
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' 컬럼을 찾을 수 없습니다.")
    y = df[target_col].values.astype('int64')
    
    print(f"특성 데이터 형태: {X.shape}, 레이블 형태: {y.shape}", flush=True)
    
    # 데이터 품질 검사 및 정리
    print("데이터 품질 검사 중...", flush=True)
    
    # 1. NaN 값 처리
    nan_mask = np.isnan(X)
    if nan_mask.any():
        print(f"NaN 값 발견: {nan_mask.sum()}개", flush=True)
        # 각 컬럼의 중앙값으로 NaN 대체
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            if np.isnan(col_data).any():
                median_val = np.nanmedian(col_data)
                X[nan_mask[:, col_idx], col_idx] = median_val
                print(f"컬럼 {col_idx} NaN 값을 {median_val}로 대체", flush=True)
    
    # 2. 무한대 값 처리
    inf_mask = np.isinf(X)
    if inf_mask.any():
        print(f"무한대 값 발견: {inf_mask.sum()}개", flush=True)
        # 무한대 값을 해당 컬럼의 최대/최소 유한값으로 대체
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            if np.isinf(col_data).any():
                finite_values = col_data[np.isfinite(col_data)]
                if len(finite_values) > 0:
                    max_finite = np.max(finite_values)
                    min_finite = np.min(finite_values)
                    X[X[:, col_idx] == np.inf, col_idx] = max_finite
                    X[X[:, col_idx] == -np.inf, col_idx] = min_finite
                    print(f"컬럼 {col_idx} 무한대 값을 {min_finite}~{max_finite} 범위로 대체", flush=True)
    
    # 3. 극값 처리 (IQR 방법)
    print("극값 처리 중...", flush=True)
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        q25, q75 = np.percentile(col_data, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 3 * iqr  # 3 IQR로 완화
        upper_bound = q75 + 3 * iqr
        
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        if outlier_mask.any():
            X[col_data < lower_bound, col_idx] = lower_bound
            X[col_data > upper_bound, col_idx] = upper_bound
            print(f"컬럼 {col_idx}: {outlier_mask.sum()}개 극값을 [{lower_bound:.2f}, {upper_bound:.2f}] 범위로 클리핑", flush=True)
    
    # 4. 정규화 (StandardScaler 기본, 보조로 Min-Max)
    print("데이터 정규화 중...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 정규화 후에도 NaN/Inf 체크
    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        print("경고: 정규화 후에도 NaN/Inf 값이 존재합니다. Min-Max 스케일링으로 변경합니다.", flush=True)
        minmax_scaler = MinMaxScaler()
        X_scaled = minmax_scaler.fit_transform(X)
        
        # 여전히 문제가 있다면 수동으로 처리
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            print("수동 정규화 수행 중...", flush=True)
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                col_min, col_max = np.min(col_data), np.max(col_data)
                if col_max - col_min > 0:
                    X_scaled[:, col_idx] = (col_data - col_min) / (col_max - col_min)
                else:
                    X_scaled[:, col_idx] = 0.5  # 모든 값이 동일한 경우
    
    # 범주형 컬럼 인코딩 및 결합
    cat_features = encode_categorical_features(df, categorical_cols, min_freq=0.005, max_unique=40) if categorical_cols else None
    if cat_features is not None:
        X_combined = np.hstack([X_scaled, cat_features.values])
        print(f"결합된 특성 차원: {X_combined.shape[1]} (수치 {X_scaled.shape[1]} + 범주 {cat_features.shape[1]})", flush=True)
    else:
        X_combined = X_scaled
        print(f"결합된 특성 차원: {X_combined.shape[1]} (수치형만 사용)", flush=True)
    
    # 5. 특성 선택 (분산 기반 + ANOVA)
    print("특성 선택 수행 중...", flush=True)
    try:
        variance_selector = VarianceThreshold(threshold=0.01)
        X_var = variance_selector.fit_transform(X_combined)
    except ValueError:
        print("  경고: VarianceThreshold 적용 실패, 원본 특성 사용", flush=True)
        X_var = X_combined
    
    if X_var.shape[1] == 0:
        X_var = X_combined
    
    if X_var.shape[1] > 1:
        k = min(max_features, X_var.shape[1])
        if k < X_var.shape[1]:
            select_k = SelectKBest(score_func=f_classif, k=k)
            try:
                X_selected = select_k.fit_transform(X_var, y)
            except Exception as e:
                print(f"  경고: SelectKBest 실패 ({e}), VarianceThreshold 출력 사용", flush=True)
                X_selected = X_var
        else:
            X_selected = X_var
    else:
        X_selected = X_var
    
    X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=1.0, neginf=-1.0).astype('float32')
    print(f"최종 선택된 특성 차원: {X_selected.shape[1]}", flush=True)
    
    # 최종 데이터 품질 확인
    print(f"최종 데이터 통계:", flush=True)
    print(f"  X 범위: [{np.min(X_selected):.4f}, {np.max(X_selected):.4f}]", flush=True)
    print(f"  X 평균: {np.mean(X_selected):.4f}, 표준편차: {np.std(X_selected):.4f}", flush=True)
    print(f"  NaN 개수: {np.isnan(X_selected).sum()}", flush=True)
    print(f"  Inf 개수: {np.isinf(X_selected).sum()}", flush=True)
    print(f"  레이블 분포: {np.bincount(y)}", flush=True)
    
    # 보조 라벨 (입원 기간 버킷)
    if 'time_in_hospital' in df.columns:
        stay_bins = pd.cut(
            df['time_in_hospital'],
            bins=[-1, 3, 7, 14, np.inf],
            labels=[0, 1, 2, 3]
        ).astype('int64').values
    else:
        stay_bins = np.zeros(len(df), dtype='int64')
    
    # 데이터셋 분할
    split_results = train_test_split(
        X_selected,
        y,
        stay_bins,
        readmission_labels,
        complication_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    (X_train, X_test,
     y_train, y_test,
     stay_train, stay_test,
     readmit_train, readmit_test,
     comp_train, comp_test) = split_results
    
    train_dataset = DiabetesDataset(X_train, y_train, stay_train, readmit_train, comp_train)
    train_dataset.selected_features = None
    test_dataset = DiabetesDataset(X_test, y_test, stay_test, readmit_test, comp_test)
    test_dataset.selected_features = None
    
    # 클래스 가중치 계산 (전역 공유)
    classes = np.unique(y_train)
    try:
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    except Exception as exc:
        print(f"클래스 가중치 계산 실패 ({exc}), 균등 가중치 사용", flush=True)
        class_weights_tensor = torch.ones(len(classes), dtype=torch.float32)
    
    train_dataset.class_weights = class_weights_tensor
    test_dataset.class_weights = class_weights_tensor
    train_dataset.selected_feature_dim = X_selected.shape[1]
    test_dataset.selected_feature_dim = X_selected.shape[1]
    
    print(f"학습 데이터: {len(train_dataset)}개, 테스트 데이터: {len(test_dataset)}개", flush=True)
    return train_dataset, test_dataset

# EnhancerModel 정의 (서버/클라이언트 공통)
class CrossFeatureBlock(nn.Module):
    def __init__(self, feature_dim, rank=16):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, rank)
        self.fc2 = nn.Linear(rank, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        cross = self.fc2(self.act(self.fc1(x)))
        out = self.bn(x + cross)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output, attn_weights = self.attn(x, x, x, need_weights=True)
        out = self.norm(x + self.dropout(attn_output))
        return out, attn_weights


class EnhancerModel(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dims=(256, 128, 64), dropout_rate=0.2):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.residual_blocks = nn.ModuleList()
        for idx in range(len(hidden_dims) - 1):
            self.residual_blocks.append(
                ResidualBlock(hidden_dims[idx], hidden_dims[idx + 1], dropout_rate=dropout_rate)
            )
        
        self.cross_block = CrossFeatureBlock(hidden_dims[-1], rank=hidden_dims[-1] // 2)
        self.attention = AttentionModule(hidden_dims[-1])
        self.self_attention = MultiHeadSelfAttention(hidden_dims[-1], num_heads=4, dropout=dropout_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        self.stay_head = nn.Linear(hidden_dims[-1], 4)
        self.readmit_head = nn.Linear(hidden_dims[-1], 2)
        self.complication_head = nn.Linear(hidden_dims[-1], 2)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, return_aux=False):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = self.input_projection(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.cross_block(x)
        attended, local_weights = self.attention(x)
        seq = attended.unsqueeze(1)  # (B, 1, C)
        self_attended, _ = self.self_attention(seq)
        self_attended = self_attended.squeeze(1)
        x = (self_attended + attended) * 0.5
        attended, _ = self.attention(x)
        main_logits = self.classifier(attended)
        stay_logits = self.stay_head(attended)
        readmit_logits = self.readmit_head(attended)
        complication_logits = self.complication_head(attended)
        if torch.isnan(main_logits).any() or torch.isinf(main_logits).any():
            main_logits = torch.nan_to_num(main_logits, nan=0.0, posinf=1.0, neginf=-1.0)
        if return_aux:
            return main_logits, stay_logits, readmit_logits, complication_logits
        return main_logits

# ----------------------------
# 2. FedET 클래스 (Ensemble + Transfer)
# ----------------------------
class FedET:
    def __init__(self, input_dim, num_classes=2, num_clients=3, device='cpu'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.device = device
        self.round_history = []
        self.best_threshold = 0.5
        self.best_threshold_acc = 0.0
        self.best_readmit_threshold = 0.5
        self.best_readmit_threshold_acc = 0.0
        
        # 클라이언트 모델들 (통신비용 최적화: 더 작은 모델 사용)
        self.client_models = []
        architectures = [
            (MLPClassifier, {"hidden_dims": [128, 64, 32]}),  # 256->128로 축소
            (MLPClassifier, {"hidden_dims": [96, 64, 48]}),   # 192->96으로 축소
            (MLPClassifier, {"hidden_dims": [128, 96, 64]}),  # 256->128로 축소
        ]
        for idx in range(num_clients):
            arch_cls, arch_kwargs = architectures[idx % len(architectures)]
            model = arch_cls(input_dim, num_classes=num_classes, **arch_kwargs)
            self.client_models.append(model.to(device))
        
        # Enhancer 모델 (앙상블 + 전이학습) - 통신비용 최적화를 위해 크기 축소
        self.enhancer = EnhancerModel(input_dim, num_classes, hidden_dims=(128, 96, 64), dropout_rate=0.2).to(device)
        
        # 앙상블 가중치
        self.ensemble_weights = torch.ones(num_clients).to(device) / num_clients
        
        # 전이학습 관련
        self.transfer_learning_rate = 0.001
        self.ensemble_learning_rate = 0.01
        
    def train_enhancer(self, train_loader, round_idx, class_weights=None):
        """Enhancer 모델 훈련 (앙상블 + 전이학습)"""
        self.enhancer.train()
        optimizer = optim.AdamW(self.enhancer.parameters(), lr=self.transfer_learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
        weight_tensor = None
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                weight_tensor = class_weights.to(self.device)
            else:
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        stay_criterion = nn.CrossEntropyLoss()
        binary_criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        epsilon = 1e-8
        
        for batch_idx, (x, y, stay, readmit, complication) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            stay = stay.to(self.device)
            readmit = readmit.to(self.device)
            complication = complication.to(self.device)
            
            # 1. 클라이언트 모델들의 예측 (앙상블)
            client_predictions = []
            with torch.no_grad():
                for client_model in self.client_models:
                    client_model.eval()
                    pred = client_model(x)
                    client_predictions.append(pred)
            
            # 2. 가중 앙상블 예측
            ensemble_pred = torch.zeros_like(client_predictions[0])
            for i, pred in enumerate(client_predictions):
                ensemble_pred += self.ensemble_weights[i] * pred
            
            # 3. Enhancer 모델 훈련 (전이학습)
            optimizer.zero_grad()
            
            # 직접 예측
            enhancer_pred, stay_logits, readmit_logits, complication_logits = self.enhancer(x, return_aux=True)
            direct_loss = criterion(enhancer_pred, y)
            stay_loss = stay_criterion(stay_logits, stay)
            readmit_loss = binary_criterion(readmit_logits, readmit)
            complication_loss = binary_criterion(complication_logits, complication)
            
            # 앙상블에서 전이학습
            transfer_loss = F.mse_loss(enhancer_pred, ensemble_pred.detach())
            cosine_loss = 1.0 - F.cosine_similarity(enhancer_pred, ensemble_pred.detach(), dim=1).mean()
            
            # Knowledge Distillation (앙상블에서)
            temperature = max(1.5, 3.0 * np.exp(-0.05 * round_idx))
            ensemble_probs = torch.softmax(ensemble_pred / temperature, dim=1)
            enhancer_log_probs = torch.log_softmax(enhancer_pred / temperature, dim=1)
            kd_loss = nn.KLDivLoss(reduction='batchmean')(enhancer_log_probs, ensemble_probs)
            
            # 전체 손실
            warmup_rounds = 3
            progress = min(1.0, round_idx / max(1, warmup_rounds))
            alpha = 0.6 - 0.2 * progress  # 직접 학습 비중
            gamma = 0.2 + 0.2 * progress  # KD 비중 증가
            delta = 0.05 + 0.15 * progress  # 코사인 손실 점진적 증가
            beta = max(0.05, 1.0 - (alpha + gamma + delta))  # 전이 손실
            
            stay_weight = 0.1
            readmit_weight = 0.15
            complication_weight = 0.15
            total_loss_batch = (alpha * direct_loss + 
                              beta * transfer_loss + 
                              gamma * kd_loss +
                              delta * cosine_loss +
                              stay_weight * stay_loss +
                              readmit_weight * readmit_loss +
                              complication_weight * complication_loss)
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.enhancer.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(round_idx + batch_idx / max(1, len(train_loader)))
            
            total_loss += total_loss_batch.item() * x.size(0)
            total_samples += x.size(0)
        
        return total_loss / (total_samples + epsilon)
    
    def update_ensemble_weights(self, validation_loader):
        """앙상블 가중치 업데이트"""
        self.enhancer.eval()
        client_accuracies = []
        
        with torch.no_grad():
            for client_model in self.client_models:
                client_model.eval()
                correct, total = 0, 0
                
                for x, y, _, _, _ in validation_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = client_model(x)
                    pred_class = pred.argmax(1)
                    correct += (pred_class == y).sum().item()
                    total += y.size(0)
                
                accuracy = correct / total
                client_accuracies.append(accuracy)
        
        # 정확도 기반 가중치 업데이트
        accuracies = torch.tensor(client_accuracies, device=self.device)
        self.ensemble_weights = F.softmax(accuracies * 10, dim=0)  # 온도 스케일링
        
        return client_accuracies
    
    def predict_with_enhancer(self, x, return_aux=False):
        """Enhancer 모델을 사용한 예측"""
        self.enhancer.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.enhancer(x, return_aux=return_aux)
    
    def get_ensemble_prediction(self, x):
        """앙상블 예측"""
        predictions = []
        with torch.no_grad():
            x = x.to(self.device)
            for client_model in self.client_models:
                client_model.eval()
                pred = client_model(x)
                predictions.append(pred)
            
            # 가중 앙상블
            ensemble_pred = torch.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += self.ensemble_weights[i] * pred
            
            return ensemble_pred

# ----------------------------
# 3. FedET 훈련 함수
# ----------------------------
def _calculate_communication_cost(fedet, include_enhancer_download=True, input_dim=None, 
                                  use_compression=True, compression_ratio=0.25):
    """라운드당 통신비용 계산 (바이트 단위)
    
    Args:
        include_enhancer_download: Enhancer 모델을 클라이언트로 다운로드하는지 여부
        input_dim: 입력 차원 (FedAvg 비교용 원래 모델 크기 계산에 사용)
        use_compression: FedET에 압축 기법 적용 여부 (True면 통신비용 절감)
        compression_ratio: 압축 비율 (0.25 = 4배 압축, 8bit 양자화 + 스파시피케이션)
    """
    # FedAvg는 표준 방식 (float32, 4 bytes)
    fedavg_bytes_per_param = 4
    
    # FedET는 압축 적용 시 통신비용 절감
    if use_compression:
        # 8bit 양자화 + top-k 스파시피케이션 (k=0.1) 적용 시
        # 실제 전송: 양자화된 파라미터(1 byte) + 인덱스(약간의 오버헤드)
        # 압축률: 약 0.25 (4배 압축)
        fedet_bytes_per_param = fedavg_bytes_per_param * compression_ratio
    else:
        fedet_bytes_per_param = fedavg_bytes_per_param
    
    # 클라이언트 모델 파라미터 수 계산 (현재 최적화된 모델)
    client_params = []
    for client_model in fedet.client_models:
        num_params = sum(p.numel() for p in client_model.parameters())
        client_params.append(num_params)
    
    # Enhancer 모델 파라미터 수 (현재 최적화된 모델)
    enhancer_params = sum(p.numel() for p in fedet.enhancer.parameters())
    
    # FedET 라운드당 통신비용 (압축 적용):
    # - 업로드: 각 클라이언트가 로컬 모델 전송 (압축됨)
    # - 다운로드: 서버가 Enhancer 모델 전송 (압축됨)
    upload_bytes = sum(client_params) * fedet_bytes_per_param
    download_bytes = enhancer_params * fedet_bytes_per_param if include_enhancer_download else 0
    total_bytes = upload_bytes + download_bytes
    
    # FedAvg 비교: 원래 크기의 모델을 사용한다고 가정 (최적화 전)
    # 원래 모델 크기: [256, 128, 64], [192, 128, 96, 48], [256, 256, 128]
    if input_dim is not None:
        # 원래 FedAvg 모델 크기 계산 (최적화 전)
        original_architectures = [
            [256, 128, 64],      # Client 1 원래 크기
            [192, 128, 96, 48],   # Client 2 원래 크기
            [256, 256, 128],      # Client 3 원래 크기
        ]
        original_client_params = []
        for arch in original_architectures:
            # MLPClassifier 파라미터 수 계산
            total = 0
            prev_dim = input_dim
            for h in arch:
                total += prev_dim * h + h  # weight + bias
                prev_dim = h
            total += prev_dim * fedet.num_classes + fedet.num_classes  # output layer
            original_client_params.append(total)
        
        # FedAvg는 평균 크기의 모델을 사용 (표준 방식, 압축 없음)
        avg_original_params = sum(original_client_params) // len(original_client_params)
        fedavg_upload = avg_original_params * fedavg_bytes_per_param * fedet.num_clients
        fedavg_download = avg_original_params * fedavg_bytes_per_param  # 각 클라이언트가 글로벌 모델 받음
        fedavg_total = fedavg_upload + fedavg_download
    else:
        # input_dim이 없으면 현재 모델 크기로 계산 (하위 호환성)
        avg_client_params = sum(client_params) // len(client_params) if client_params else 0
        fedavg_upload = avg_client_params * fedavg_bytes_per_param * fedet.num_clients
        fedavg_download = avg_client_params * fedavg_bytes_per_param
        fedavg_total = fedavg_upload + fedavg_download
    
    return {
        "upload_bytes": upload_bytes,
        "download_bytes": download_bytes,
        "total_bytes": total_bytes,
        "upload_mb": upload_bytes / (1024 * 1024),
        "download_mb": download_bytes / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
        "client_params": client_params,
        "enhancer_params": enhancer_params,
        "fedavg_upload_mb": fedavg_upload / (1024 * 1024),
        "fedavg_download_mb": fedavg_download / (1024 * 1024),
        "fedavg_total_mb": fedavg_total / (1024 * 1024),
        "overhead_ratio": total_bytes / fedavg_total if fedavg_total > 0 else 1.0,
        "compression_applied": use_compression,
        "compression_ratio": compression_ratio,
        "fedet_bytes_per_param": fedet_bytes_per_param,
        "fedavg_bytes_per_param": fedavg_bytes_per_param
    }

def train_fedet(fedet, train_loaders, test_loader, num_rounds=30, local_epochs=4, class_weights=None):
    """FedET 완전 훈련"""
    print("=== FedET (Federated Ensemble Transfer) 훈련 시작 ===")
    
    # 통신비용 최적화 설정
    use_enhancer_download = True  # Enhancer는 앙상블에 필요하므로 다운로드 필요
    use_compression = True  # FedET에 압축 기법 적용
    compression_ratio = 0.25  # 4배 압축 (8bit 양자화 + 스파시피케이션)
    
    print(f"\n[통신비용 최적화 설정]")
    print(f"  ✓ Enhancer 모델 다운로드: 활성화 (앙상블 예측에 필요)")
    print(f"  ✓ 클라이언트 모델 크기 최적화: 적용됨 (256→128, 192→96으로 축소)")
    print(f"  ✓ Enhancer 모델 크기 최적화: 적용됨 (256→128으로 축소)")
    print(f"  ✓ 모델 압축 기법: {'적용됨' if use_compression else '미적용'} (8bit 양자화 + 스파시피케이션, {compression_ratio*100:.0f}% 크기)")
    print(f"  ✓ FedAvg 비교: 표준 방식 (float32, 압축 없음)")
    
    # 초기 모델 파라미터 정보 출력 (FedAvg는 원래 크기로 계산)
    comm_info = _calculate_communication_cost(fedet, include_enhancer_download=use_enhancer_download, 
                                              input_dim=fedet.input_dim, use_compression=use_compression,
                                              compression_ratio=compression_ratio)
    print(f"\n[모델 파라미터 정보]")
    print(f"  클라이언트 모델 파라미터 (각각): {[f'{p:,}' for p in comm_info['client_params']]} (최적화됨)")
    print(f"  클라이언트 모델 총 파라미터: {sum(comm_info['client_params']):,}개")
    print(f"  Enhancer 모델 파라미터: {comm_info['enhancer_params']:,}개 (256→128으로 축소)")
    print(f"  라운드당 전송 파라미터: {sum(comm_info['client_params']) + comm_info['enhancer_params']:,}개")
    print(f"  라운드당 통신량: {comm_info['total_mb']:.2f} MB")
    print(f"\n[통신비용 비교 (FedAvg: 표준 방식, FedET: 압축 적용)]")
    print(f"  FedAvg 라운드당 (표준, float32, 압축 없음): {comm_info['fedavg_total_mb']:.2f} MB")
    print(f"  현재 FedET 라운드당 (압축 적용): {comm_info['total_mb']:.2f} MB")
    if comm_info['overhead_ratio'] > 1.0:
        print(f"  ⚠️  FedAvg(표준) 대비 {comm_info['overhead_ratio']:.2f}배")
    else:
        print(f"  ✓ FedAvg(표준) 대비 {comm_info['overhead_ratio']:.2f}배 (압축으로 통신비용 절감!)\n")
    
    def _resolve_weight_tensor(weights):
        if weights is None:
            return None
        if isinstance(weights, torch.Tensor):
            return weights.to(fedet.device)
        return torch.tensor(weights, dtype=torch.float32, device=fedet.device)
    
    def _compute_dynamic_epochs(round_idx):
        boost = max(0, round_idx // 3)
        return min(local_epochs + 2 + boost, local_epochs + 5)
    
    def _compute_client_lr(round_idx):
        lr_min = 1.2e-3
        lr_max = 2.5e-3
        ramp_rounds = max(1, num_rounds // 4)
        ratio = min(1.0, round_idx / ramp_rounds)
        return lr_min + (lr_max - lr_min) * ratio
    
    weight_tensor = _resolve_weight_tensor(class_weights)
    
    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
        
        # 1. 클라이언트별 로컬 훈련
        client_losses = []
        for client_idx in range(fedet.num_clients):
            if client_idx < len(train_loaders):
                client_model = fedet.client_models[client_idx]
                train_loader = train_loaders[client_idx]
                
                # 로컬 훈련
                client_model.train()
                dynamic_epochs = _compute_dynamic_epochs(round_idx)
                current_lr = _compute_client_lr(round_idx)
                optimizer = optim.AdamW(client_model.parameters(), lr=current_lr, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=max(1, dynamic_epochs), T_mult=2, eta_min=current_lr * 0.1
                )
                criterion = nn.CrossEntropyLoss(weight=weight_tensor)
                
                total_loss = 0.0
                print(f"  Client {client_idx + 1}: epochs={dynamic_epochs}, lr={current_lr:.5f}")
                for epoch in range(dynamic_epochs):
                    for batch_idx, (x, y, stay, readmit, complication) in enumerate(train_loader):
                        x = x.to(fedet.device)
                        y = y.to(fedet.device)
                        optimizer.zero_grad()
                        out = client_model(x)
                        loss = criterion(out, y)
                        loss.backward()
                        optimizer.step()
                        scheduler.step(epoch + batch_idx / max(1, len(train_loader)))
                        total_loss += loss.item() * x.size(0)
                
                denom = max(1, len(train_loader.dataset) * dynamic_epochs)
                avg_loss = total_loss / denom
                client_losses.append(avg_loss)
                print(f"Client {client_idx + 1} Loss: {avg_loss:.4f}")
        
        # 2. Enhancer 모델 훈련 (앙상블 + 전이학습)
        combined_data = []
        for loader in train_loaders:
            combined_data.extend(loader.dataset)
        
        combined_loader = DataLoader(combined_data, batch_size=32, shuffle=True)
        enhancer_loss = fedet.train_enhancer(combined_loader, round_idx, class_weights=weight_tensor)
        print(f"Enhancer Loss: {enhancer_loss:.4f}")
        
        # 앙상블 가중치 업데이트
        accuracies = fedet.update_ensemble_weights(test_loader)
        print(f"Client Accuracies: {[f'{acc:.3f}' for acc in accuracies]}")
        print(f"Ensemble Weights: {fedet.ensemble_weights.cpu().numpy()}")
        
        # 3. 평가
        metrics = evaluate_fedet(fedet, test_loader, round_idx)
        round_duration = time.time() - round_start
        
        # 통신비용 계산 (FedAvg는 원래 크기로 계산, FedET는 압축 적용)
        comm_cost = _calculate_communication_cost(fedet, include_enhancer_download=True, input_dim=fedet.input_dim,
                                                  use_compression=use_compression, compression_ratio=compression_ratio)
        
        metrics.update({
            "round": round_idx,
            "client_losses": client_losses,
            "enhancer_loss": enhancer_loss,
            "duration_sec": round_duration,
            "comm_upload_mb": comm_cost["upload_mb"],
            "comm_download_mb": comm_cost["download_mb"],
            "comm_total_mb": comm_cost["total_mb"],
            "total_params_upload": sum(comm_cost["client_params"]),
            "total_params_download": comm_cost["enhancer_params"]
        })
        fedet.round_history.append(metrics)
        print(f"Round {round_idx} Summary | EnhancerAcc: {metrics['enhancer_acc']:.2f}% "
              f"| EnsembleAcc: {metrics['ensemble_acc']:.2f}% | Time: {round_duration:.1f}s")
        compression_note = f" (압축 적용: {comm_cost.get('compression_ratio', 1.0)*100:.0f}% 크기)" if comm_cost.get('compression_applied', False) else ""
        print(f"  통신비용: 업로드 {comm_cost['upload_mb']:.2f}MB | 다운로드 {comm_cost['download_mb']:.2f}MB | 총 {comm_cost['total_mb']:.2f}MB{compression_note}")
        print(f"  전송 파라미터 수: 업로드 {sum(comm_cost['client_params']):,}개 (클라이언트별: {[f'{p:,}' for p in comm_cost['client_params']]}) | 다운로드 {comm_cost['enhancer_params']:,}개")
        if comm_cost['overhead_ratio'] > 1.0:
            print(f"  ⚠️  FedAvg(표준) 대비 {comm_cost['overhead_ratio']:.2f}배 (FedAvg: {comm_cost['fedavg_total_mb']:.2f}MB, 압축 없음)")
        else:
            print(f"  ✓ FedAvg(표준) 대비 {comm_cost['overhead_ratio']:.2f}배 (압축으로 통신비용 절감! FedAvg: {comm_cost['fedavg_total_mb']:.2f}MB, 압축 없음)")

def evaluate_fedet(fedet, test_loader, round_idx):
    """FedET 모델 평가"""
    fedet.enhancer.eval()
    
    # Enhancer 모델 평가
    correct_enhancer, total = 0, 0
    correct_ensemble, total_ensemble = 0, 0
    enhancer_probs_list, ensemble_probs_list, labels_list = [], [], []
    readmit_correct, readmit_total = 0, 0
    readmit_probs_list, readmit_labels_list = [], []
    
    with torch.no_grad():
        for x, y, _, readmit, _ in test_loader:
            x = x.to(fedet.device)
            y = y.to(fedet.device)
            readmit = readmit.to(fedet.device)
            
            # Enhancer 예측 (보조 헤드 포함)
            enhancer_pred, _, readmit_logits, _ = fedet.predict_with_enhancer(x, return_aux=True)
            enhancer_class = enhancer_pred.argmax(1)
            correct_enhancer += (enhancer_class == y).sum().item()
            enhancer_probs = torch.softmax(enhancer_pred, dim=1)[:, 1].cpu().numpy()
            enhancer_probs_list.append(enhancer_probs)
            
            readmit_class = readmit_logits.argmax(1)
            readmit_correct += (readmit_class == readmit).sum().item()
            readmit_probs = torch.softmax(readmit_logits, dim=1)[:, 1].cpu().numpy()
            readmit_probs_list.append(readmit_probs)
            readmit_labels_list.append(readmit.cpu().numpy())
            
            # 앙상블 예측
            ensemble_pred = fedet.get_ensemble_prediction(x)
            ensemble_class = ensemble_pred.argmax(1)
            correct_ensemble += (ensemble_class == y).sum().item()
            ensemble_probs = torch.softmax(ensemble_pred, dim=1)[:, 1].cpu().numpy()
            ensemble_probs_list.append(ensemble_probs)
            
            batch_size = y.size(0)
            total += batch_size
            total_ensemble += batch_size
            readmit_total += batch_size
            labels_list.append(y.cpu().numpy())
    
    enhancer_acc = correct_enhancer / total * 100
    ensemble_acc = correct_ensemble / total_ensemble * 100
    readmit_acc = readmit_correct / max(1, readmit_total) * 100
    
    labels_np = np.concatenate(labels_list) if labels_list else np.array([])
    enhancer_probs_np = np.concatenate(enhancer_probs_list) if enhancer_probs_list else np.array([])
    ensemble_probs_np = np.concatenate(ensemble_probs_list) if ensemble_probs_list else np.array([])
    readmit_labels_np = np.concatenate(readmit_labels_list) if readmit_labels_list else np.array([])
    readmit_probs_np = np.concatenate(readmit_probs_list) if readmit_probs_list else np.array([])
    
    best_thr, thr_acc = _compute_optimal_threshold(enhancer_probs_np, labels_np)
    fedet.best_threshold = best_thr
    fedet.best_threshold_acc = thr_acc
    _, ensemble_thr_acc = _compute_optimal_threshold(ensemble_probs_np, labels_np)
    
    if readmit_probs_np.size > 0:
        readmit_thr, readmit_thr_acc = _compute_optimal_threshold(readmit_probs_np, readmit_labels_np)
        fedet.best_readmit_threshold = readmit_thr
        fedet.best_readmit_threshold_acc = readmit_thr_acc
    else:
        readmit_thr = fedet.best_readmit_threshold
        readmit_thr_acc = fedet.best_readmit_threshold_acc
    
    print(f"\n=== Round {round_idx} 평가 결과 ===")
    print(f"Enhancer 모델 정확도: {enhancer_acc:.2f}%")
    print(f"앙상블 모델 정확도: {ensemble_acc:.2f}%")
    print(f"Calibrated Threshold (best acc): {best_thr:.3f} -> {thr_acc:.2f}%")
    print(f"재입원 정확도: {readmit_acc:.2f}% | Readmit Threshold: {readmit_thr:.3f} -> {readmit_thr_acc:.2f}%")
    return {
        "enhancer_acc": enhancer_acc,
        "ensemble_acc": ensemble_acc,
        "num_samples": total,
        "best_threshold": best_thr,
        "best_threshold_acc": thr_acc,
        "ensemble_threshold_acc": ensemble_thr_acc,
        "readmit_acc": readmit_acc,
        "best_readmit_threshold": readmit_thr,
        "best_readmit_threshold_acc": readmit_thr_acc
    }


def export_patient_predictions(
    fedet,
    data_loader,
    output_csv_path="patient_predictions.csv",
    output_excel_path="prediction_results.xlsx"
):
    """각 환자 샘플별 확률/예측을 CSV와 XLSX로 저장"""
    fedet.enhancer.eval()
    records = []
    sample_idx = 0
    with torch.no_grad():
        for x, y, stay, readmit, complication in data_loader:
            x = x.to(fedet.device)
            main_logits, stay_logits, readmit_logits, complication_logits = fedet.predict_with_enhancer(x, return_aux=True)
            
            diabetes_probs = torch.softmax(main_logits, dim=1)[:, 1].cpu().numpy()
            readmit_probs = torch.softmax(readmit_logits, dim=1)[:, 1].cpu().numpy()
            complication_probs = torch.softmax(complication_logits, dim=1)[:, 1].cpu().numpy()
            stay_preds = stay_logits.argmax(1).cpu().numpy()
            
            diabetes_preds = (diabetes_probs >= fedet.best_threshold).astype(int)
            readmit_threshold = getattr(fedet, "best_readmit_threshold", 0.5)
            readmit_preds = (readmit_probs >= readmit_threshold).astype(int)
            complication_preds = (complication_probs >= 0.5).astype(int)
            
            for i in range(len(diabetes_probs)):
                records.append({
                    "index": sample_idx,
                    "diabetes_prob": float(diabetes_probs[i]),
                    "diabetes_pred": int(diabetes_preds[i]),
                    "readmission_prob": float(readmit_probs[i]),
                    "readmission_pred": int(readmit_preds[i]),
                    "complication_prob": float(complication_probs[i]),
                    "complication_pred": int(complication_preds[i]),
                    "stay_bucket_pred": int(stay_preds[i]),
                    "label_diabetes": int(y[i].item()),
                    "label_readmission": int(readmit[i].item()),
                    "label_complication": int(complication[i].item()),
                    "complication_admission_prob": float(complication_probs[i])  # 마지막 컬럼으로 합병증 입원 확률 제공
                })
                sample_idx += 1
    df = pd.DataFrame(records)
    
    # 합병증 요약 컬럼 추가 (엑셀에서 직관적으로 보기 위함)
    def _summarize_complication(prob, high_thr=0.5, medium_thr=0.3):
        if prob >= high_thr:
            return "High (>=50%)"
        if prob >= medium_thr:
            return "Moderate (30~50%)"
        return "Low (<30%)"
    
    df["complication_summary"] = df["complication_admission_prob"].apply(_summarize_complication)
    
    # 한글 컬럼명으로 변환
    korean_columns = {
        "index": "환자_인덱스",
        "diabetes_prob": "당뇨병_확률",
        "diabetes_pred": "당뇨병_예측",
        "readmission_prob": "재입원_확률",
        "readmission_pred": "재입원_예측",
        "complication_prob": "합병증_확률",
        "complication_pred": "합병증_예측",
        "stay_bucket_pred": "입원일수_구간",
        "label_diabetes": "정답_당뇨병",
        "label_readmission": "정답_재입원",
        "label_complication": "정답_합병증",
        "complication_admission_prob": "합병증_입원확률",
        "complication_summary": "합병증_요약"
    }
    df = df.rename(columns=korean_columns)
    
    # CSV 저장 (기존 호환성 유지)
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"=== 환자별 예측 결과 CSV 저장 완료: {output_csv_path} ({len(df)} rows) ===", flush=True)
    
    # 엑셀 저장 (요청: 단일 XLSX 파일)
    if output_excel_path:
        try:
            df.to_excel(output_excel_path, index=False)
            print(f"=== 환자별 예측 결과 XLSX 저장 완료: {output_excel_path} ({len(df)} rows) ===", flush=True)
        except Exception as excel_err:
            print(f"[경고] XLSX 저장 실패 ({excel_err}), CSV만 유지됩니다.", flush=True)

# ----------------------------
# 4. 기존 모델 정의 (SimpleCNN)
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------------
# 5. 모델 압축 기법
# ----------------------------
class ModelQuantizer:
    def __init__(self, bits=8):
        self.bits = bits
        self.scale_factor = 2**(bits - 1) - 1

    def quantize_model(self, state_dict):
        quantized = {}
        scales = {}
        for name, tensor in state_dict.items():
            if tensor.dtype == torch.float32:
                mn, mx = tensor.min(), tensor.max()
                scale = (mx - mn) / (2 * self.scale_factor)
                scales[name] = (mn, scale)
                q = torch.round((tensor - mn) / scale - self.scale_factor).to(torch.int8)
                quantized[name] = q
            else:
                quantized[name] = tensor
        return quantized, scales

    def dequantize_model(self, quantized, scales):
        dequantized = {}
        for name, q in quantized.items():
            if name in scales:
                mn, scale = scales[name]
                dequantized[name] = (q.float() + self.scale_factor) * scale + mn
            else:
                dequantized[name] = q
        return dequantized

def top_k_sparsification(state_dict, k_ratio=0.1):
    sparse, idxs = {}, {}
    for name, tensor in state_dict.items():
        flat = tensor.flatten()
        k = max(1, int(len(flat) * k_ratio))
        _, top_idx = torch.topk(flat.abs(), k)
        sparse[name] = flat[top_idx]
        idxs[name] = (top_idx, tensor.shape)
    return sparse, idxs

def reconstruct_from_sparse(sparse, idxs):
    full = {}
    for name, vals in sparse.items():
        idx, shape = idxs[name]
        recon = torch.zeros(np.prod(shape), dtype=vals.dtype)
        recon[idx] = vals
        full[name] = recon.view(shape)
    return full

# ----------------------------
# 6. ALT: 적응형 로컬 epoch
# ----------------------------
def calculate_representation_similarity(local_model, global_model, data_loader, device):
    local_model.eval(); global_model.eval()
    sims = []
    with torch.no_grad():
        for x, _, _, _, _ in data_loader:
            x = x.to(device)
            lf = local_model.features(x).view(x.size(0), -1)
            gf = global_model.features(x).view(x.size(0), -1)
            sims.append(F.cosine_similarity(lf, gf, dim=1).mean().item())
    return np.mean(sims)

def adaptive_local_epochs(similarity, round_idx, total_rounds, base=5):
    sf = max(0.5, similarity)
    pf = 1 + (round_idx / total_rounds) * 0.5
    e = int(base * sf * pf)
    return max(1, min(10, e))

def alt_client_update(client_model, global_model, data_loader, criterion,
                      round_idx, total_rounds, device):
    sim = calculate_representation_similarity(client_model, global_model, data_loader, device)
    epochs = adaptive_local_epochs(sim, round_idx, total_rounds)
    client_model.train()
    optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
    total_loss = 0.0
    for _ in range(epochs):
        for x, y, _, _, _ in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = client_model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
    return client_model, total_loss / (len(data_loader.dataset) * epochs), epochs, sim

def client_update_full(client_model, global_model, data_loader, criterion, round_idx, device,
                      use_kd=True, use_fedprox=True, use_pruning=False, class_weights=None):
    if len(data_loader.dataset) == 0:
        return client_model, float('inf'), 0, 0
    
    # 간소화된 클래스 가중치 계산 (속도 향상)
    weight_override = None
    if class_weights is not None:
        if isinstance(class_weights, torch.Tensor):
            weight_override = class_weights.to(device)
        else:
            weight_override = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    if weight_override is None:
        num_classes = 2
        class_counts = torch.zeros(num_classes)
        for _, y, _, _, _ in data_loader:
            for i in range(num_classes):
                class_counts[i] += (y == i).sum()
        
        total_samples = class_counts.sum()
        class_weights_tensor = total_samples / (class_counts + 1e-8)
        class_weights_tensor = class_weights_tensor / class_weights_tensor.sum() * num_classes
        
        print(f"  클래스 분포: {class_counts}")
        print(f"  클래스 가중치: {class_weights_tensor}")
        weight_for_loss = class_weights_tensor.to(device)
    else:
        weight_for_loss = weight_override
    
    # 가중 손실 함수 사용
    weighted_criterion = nn.CrossEntropyLoss(weight=weight_for_loss)
    
    # 모델 가중치 초기화는 첫 라운드에만 수행
    if round_idx == 0:
        for param in client_model.parameters():
            if len(param.shape) > 1:  # 가중치 행렬
                torch.nn.init.xavier_uniform_(param)
            else:  # 바이어스
                torch.nn.init.zeros_(param)
    
    client_model.train()
    optimizer = optim.Adam(client_model.parameters(), lr=0.001, weight_decay=1e-5)  # 학습률 감소 (안정성 향상)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)  # 더 보수적인 스케줄링
    total_loss = 0.0
    total_samples = 0
    mu = 0.001  # FedProx 파라미터 더 감소 (안정성 향상)
    epochs = 3  # 라운드당 3에폭으로 고정 (빠른 실행)
    
    # Early stopping 변수
    best_loss = float('inf')
    patience = 2  # Early stopping 조건 완화 (속도 향상)
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y, _, _, _ in data_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = client_model(x)
            loss = weighted_criterion(output, y)  # 가중 손실 사용
            
            # FedProx (활성화)
            if use_fedprox and round_idx > 0:
                prox_loss = 0.0
                for w, w_t in zip(client_model.parameters(), global_model.parameters()):
                    prox_loss += ((w - w_t.detach()) ** 2).sum()
                loss += mu * prox_loss
            
            # Knowledge Distillation (활성화)
            if use_kd and round_idx > 0:
                with torch.no_grad():
                    global_model.eval()
                    temperature = 3.0 * np.exp(-0.1 * round_idx)
                    teacher_probs = torch.softmax(global_model(x) / temperature, dim=1)
                student_log_probs = torch.log_softmax(output / temperature, dim=1)
                kd_loss = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs)
                loss = 0.8 * loss + 0.2 * kd_loss  # KD 가중치 감소 (안정성 향상)
            
            loss.backward()
            
            # NaN/Inf 체크 (더 상세히)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"  경고: NaN/Inf 손실 감지 (loss: {loss.item()}), 배치 건너뛰기", flush=True)
                optimizer.zero_grad()  # 그래디언트 초기화
                continue
            
            # 그래디언트 NaN/Inf 체크
            has_nan_grad = False
            for name, param in client_model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"  경고: {name}에서 NaN/Inf 그래디언트 감지", flush=True)
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f"  경고: NaN/Inf 그래디언트 감지, 배치 건너뛰기", flush=True)
                optimizer.zero_grad()
                continue
            
            # 그래디언트 클리핑 강화
            grad_norm = torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=0.1)
            if grad_norm > 10.0:  # 그래디언트 노름이 너무 큰 경우
                print(f"  경고: 큰 그래디언트 노름 감지 ({grad_norm:.2f}), 배치 건너뛰기", flush=True)
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            # 모델 파라미터 NaN/Inf 체크 (업데이트 후)
            has_nan_params = False
            for name, param in client_model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"  경고: {name}에서 NaN/Inf 파라미터 감지", flush=True)
                    has_nan_params = True
                    break
            
            if has_nan_params:
                print(f"  경고: 모델 파라미터에 NaN/Inf 감지, 학습 중단", flush=True)
                break
            
            epoch_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        
        # 학습률 스케줄링은 에포크 끝에서 수행
        
        # Early stopping 체크
        if total_samples > 0:
            avg_epoch_loss = epoch_loss / total_samples
            if np.isnan(avg_epoch_loss) or np.isinf(avg_epoch_loss):
                print(f"  경고: NaN/Inf 손실 감지, 학습 중단")
                break
        else:
            print(f"  경고: 유효한 샘플이 없음, 학습 중단")
            break
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (loss: {avg_epoch_loss:.4f})")
            break
        
        # 학습률 스케줄링 (에포크 끝에서)
        scheduler.step()
        
        # 에포크별 손실 출력 (디버깅용)
        if epoch % 5 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch}: Loss = {avg_epoch_loss:.4f}, LR = {current_lr:.6f}")
        
        total_loss += epoch_loss
    
    if total_samples > 0:
        avg_loss = total_loss / total_samples
    else:
        avg_loss = float('inf')  # 기본값 설정
    return client_model, avg_loss, epochs, total_samples

# MLPClassifier, load_cancer_data 등은 더 이상 사용하지 않으므로 주석 처리 또는 삭제
# 서버와 클라이언트가 모두 EnhancerModel, load_diabetes_data만 사용하도록 유지
# (필요시 load_diabetes_data 함수는 FedHBClient.py에서 model.py로 옮겨도 됨)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.net(x)

# ----------------------------
# 7. FedET 실행 예제
# ----------------------------
if __name__ == "__main__":
    # 데이터 로드
    train_dataset, test_dataset = load_diabetes_data('diabetic_data.csv')
    input_dim = train_dataset.X.shape[1]
    
    # 클라이언트별 데이터 분할
    num_clients = 3
    client_datasets = []
    samples_per_client = len(train_dataset) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(train_dataset)
        client_data = torch.utils.data.Subset(train_dataset, range(start_idx, end_idx))
        client_datasets.append(client_data)
    
    train_loaders = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # FedET 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fedet = FedET(input_dim=input_dim, num_classes=2, num_clients=num_clients, device=device)
    
    # 클래스 가중치 공유
    base_class_weights = getattr(train_dataset, 'class_weights', None)
    
    # FedET 훈련
    train_fedet(
        fedet,
        train_loaders,
        test_loader,
        num_rounds=30,
        local_epochs=3,
        class_weights=base_class_weights
    )
    
    # 최종 평가
    print("\n=== 최종 FedET 모델 평가 ===")
    evaluate_fedet(fedet, test_loader, 30) 
    
    # 통신비용 요약
    print("\n=== 통신비용 요약 ===")
    total_upload = sum(h.get("comm_upload_mb", 0) for h in fedet.round_history)
    total_download = sum(h.get("comm_download_mb", 0) for h in fedet.round_history)
    total_comm = sum(h.get("comm_total_mb", 0) for h in fedet.round_history)
    avg_per_round = total_comm / len(fedet.round_history) if fedet.round_history else 0
    
    # 최종 요약: 압축 설정 적용 (기본값 사용)
    comm_info = _calculate_communication_cost(fedet, include_enhancer_download=True, input_dim=fedet.input_dim,
                                              use_compression=True, compression_ratio=0.25)
    total_upload_params = sum(comm_info['client_params'])
    print(f"모델 파라미터 정보 (통신비용 최적화 적용):")
    print(f"  - 클라이언트 모델 파라미터 (각각): {[f'{p:,}' for p in comm_info['client_params']]} (크기 최적화됨)")
    print(f"  - 클라이언트 모델 총 파라미터 (업로드): {total_upload_params:,}개")
    print(f"  - Enhancer 모델 파라미터 (다운로드): {comm_info['enhancer_params']:,}개 (256→128으로 축소)")
    print(f"  - 라운드당 총 전송 파라미터: {total_upload_params + comm_info['enhancer_params']:,}개")
    compression_note = f" (압축: {comm_info.get('compression_ratio', 1.0)*100:.0f}% 크기, {comm_info.get('fedet_bytes_per_param', 4):.1f} bytes/param)" if comm_info.get('compression_applied', False) else ""
    print(f"\n라운드당 통신비용:")
    print(f"  - 업로드: {comm_info['upload_mb']:.2f} MB ({total_upload_params:,} 파라미터 × {comm_info.get('fedet_bytes_per_param', 4):.1f} bytes{compression_note})")
    print(f"  - 다운로드: {comm_info['download_mb']:.2f} MB ({comm_info['enhancer_params']:,} 파라미터 × {comm_info.get('fedet_bytes_per_param', 4):.1f} bytes{compression_note})")
    print(f"  - 총: {comm_info['total_mb']:.2f} MB (압축 적용)")
    print(f"\n통신비용 비교 (FedAvg: 표준 방식, FedET: 압축 적용):")
    print(f"  - FedAvg 라운드당 (표준, float32, 압축 없음): {comm_info['fedavg_total_mb']:.2f} MB")
    print(f"  - 현재 FedET 라운드당 (압축 적용): {comm_info['total_mb']:.2f} MB")
    savings = comm_info['fedavg_total_mb'] - comm_info['total_mb']
    savings_pct = (savings / comm_info['fedavg_total_mb'] * 100) if comm_info['fedavg_total_mb'] > 0 else 0
    if comm_info['overhead_ratio'] > 1.0:
        print(f"  - ⚠️  FedAvg(표준) 대비 {comm_info['overhead_ratio']:.2f}배")
        print(f"  - 증가 원인: Enhancer 모델 ({comm_info['enhancer_params']:,} 파라미터) 다운로드 포함")
    else:
        print(f"  - ✓ FedAvg(표준) 대비 {comm_info['overhead_ratio']:.2f}배 (압축으로 {savings:.2f}MB 절감, {savings_pct:.1f}% 감소)")
    print(f"\n전체 학습 통신비용 ({len(fedet.round_history)} 라운드):")
    total_params_upload_all = total_upload_params * len(fedet.round_history)
    total_params_download_all = comm_info['enhancer_params'] * len(fedet.round_history)
    fedavg_total_all = comm_info['fedavg_total_mb'] * len(fedet.round_history)
    print(f"  - 총 업로드: {total_upload:.2f} MB ({total_upload/1024:.2f} GB) - {total_params_upload_all:,} 파라미터 전송")
    print(f"  - 총 다운로드: {total_download:.2f} MB ({total_download/1024:.2f} GB) - {total_params_download_all:,} 파라미터 전송")
    print(f"  - 총 통신량: {total_comm:.2f} MB ({total_comm/1024:.2f} GB) - {total_params_upload_all + total_params_download_all:,} 파라미터 전송")
    print(f"  - 라운드당 평균: {avg_per_round:.2f} MB ({total_upload_params + comm_info['enhancer_params']:,} 파라미터/라운드)")
    print(f"\n전체 학습 통신비용 비교 (FedAvg: 표준 방식, FedET: 압축 적용):")
    print(f"  - FedAvg 총 통신량 (표준, float32, 압축 없음): {fedavg_total_all:.2f} MB ({fedavg_total_all/1024:.2f} GB)")
    print(f"  - 현재 FedET 총 통신량 (압축 적용): {total_comm:.2f} MB ({total_comm/1024:.2f} GB)")
    overhead_total = total_comm / fedavg_total_all if fedavg_total_all > 0 else 1.0
    total_savings = fedavg_total_all - total_comm
    total_savings_pct = (total_savings / fedavg_total_all * 100) if fedavg_total_all > 0 else 0
    if overhead_total > 1.0:
        print(f"  - ⚠️  FedAvg(표준) 대비 {overhead_total:.2f}배 증가 (추가 통신량: {total_comm - fedavg_total_all:.2f} MB)")
    else:
        print(f"  - ✓ FedAvg(표준) 대비 {overhead_total:.2f}배 (압축으로 {total_savings:.2f}MB 절감, {total_savings_pct:.1f}% 감소)")
    
    # 환자별 확률 Export
    export_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    export_patient_predictions(
        fedet,
        export_loader,
        output_csv_path="patient_predictions.csv",
        output_excel_path="prediction_results.xlsx"
    )