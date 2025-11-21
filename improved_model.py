import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight


class ImprovedDiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype("float32")
        self.y = y.astype("int64")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return x, y


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.15):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

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
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights, weights


class ImprovedEnhancerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=2,
        hidden_dims=(256, 128, 64),
        dropout_rate=0.2,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.residual_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout_rate)
            )

        self.attention = AttentionModule(hidden_dims[-1])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] // 2, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = self.feature_extractor(x)
        for block in self.residual_blocks:
            x = block(x)
        attended, _ = self.attention(x)
        out = self.classifier(attended)
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def _sanitize_numeric_array(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    if not np.isfinite(array).all():
        scaler = MinMaxScaler()
        array = scaler.fit_transform(array)
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    return array.astype("float32")


def load_improved_diabetes_data(
    csv_path,
    target_col="readmitted",
    test_size=0.2,
    random_state=42,
    max_features=12,
):
    df = pd.read_csv(csv_path)
    drop_cols = ["encounter_id", "patient_nbr"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' 컬럼을 찾을 수 없습니다.")

    df[target_col] = df[target_col].map(lambda x: 0 if x == "NO" else 1)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    if not numeric_cols:
        raise ValueError("사용 가능한 숫자형 특성이 없습니다.")

    X = df[numeric_cols].values
    y = df[target_col].values.astype("int64")
    X = _sanitize_numeric_array(X)

    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X)
    filtered_cols = [numeric_cols[i] for i in selector.get_support(indices=True)]

    k = min(max_features, X_var.shape[1])
    if k == 0:
        raise ValueError("분산 기준을 통과한 특성이 없습니다.")
    select_k = SelectKBest(score_func=f_classif, k=k)
    X_selected = select_k.fit_transform(X_var, y)
    selected_features = [filtered_cols[i] for i in select_k.get_support(indices=True)]

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = _sanitize_numeric_array(X_train_scaled)
    X_test_scaled = _sanitize_numeric_array(X_test_scaled)

    train_dataset = ImprovedDiabetesDataset(X_train_scaled, y_train)
    test_dataset = ImprovedDiabetesDataset(X_test_scaled, y_test)
    train_dataset.scaler = scaler
    test_dataset.scaler = scaler
    train_dataset.selected_features = selected_features
    test_dataset.selected_features = selected_features

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    class_weight_map = {int(cls): float(w) for cls, w in zip(classes, class_weights)}

    return (
        train_dataset,
        test_dataset,
        weights_tensor,
        selected_features,
        scaler,
        class_weight_map,
    )


def _resolve_alpha(class_weights):
    if class_weights is None:
        return 1.0
    if isinstance(class_weights, torch.Tensor):
        if class_weights.numel() > 1:
            return class_weights[-1].item()
        return class_weights.mean().item()
    if isinstance(class_weights, dict):
        return float(class_weights.get(1, 1.0))
    if isinstance(class_weights, (list, tuple)):
        return float(class_weights[-1])
    return float(class_weights)


def improved_client_update(
    client_model,
    global_model,
    data_loader,
    criterion,
    round_idx,
    device,
    class_weights=None,
    max_epochs=10,
):
    if len(data_loader.dataset) == 0:
        return client_model, float("inf"), 0, 0, 0.0

    if hasattr(client_model, "_init_weights") and round_idx == 0:
        client_model.apply(client_model._init_weights)

    focal_alpha = _resolve_alpha(class_weights)
    focal_loss = FocalLoss(alpha=focal_alpha, gamma=2.0)

    client_model.train()
    optimizer = optim.AdamW(client_model.parameters(), lr=0.0015, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-5
    )

    total_loss = 0.0
    total_samples = 0
    best_loss = float("inf")
    patience = 3
    patience_counter = 0
    trained_epochs = 0

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        samples_this_epoch = 0
        for x, y in data_loader:
            x = torch.nan_to_num(x.to(device), nan=0.0, posinf=1.0, neginf=-1.0)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = client_model(x)
            loss = focal_loss(outputs, y)

            if round_idx > 0:
                mu = 1e-3
                prox = 0.0
                for w, w_t in zip(client_model.parameters(), global_model.parameters()):
                    prox += torch.sum((w - w_t.detach()) ** 2)
                loss += mu * prox

                with torch.no_grad():
                    teacher_logits = global_model(x)
                temperature = max(1.5, 3.0 * np.exp(-0.05 * round_idx))
                kd_loss = nn.KLDivLoss(reduction="batchmean")(
                    torch.log_softmax(outputs / temperature, dim=1),
                    torch.softmax(teacher_logits / temperature, dim=1),
                )
                loss = 0.8 * loss + 0.2 * kd_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = x.size(0)
            epoch_loss += loss.item() * batch_size
            samples_this_epoch += batch_size
            total_samples += batch_size

        scheduler.step(epoch + round_idx + 1)
        trained_epochs += 1
        avg_epoch_loss = epoch_loss / max(1, samples_this_epoch)
        total_loss += epoch_loss

        if avg_epoch_loss < best_loss - 1e-4:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    avg_loss = total_loss / max(1, total_samples)

    client_model.eval()
    correct = 0
    eval_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = torch.nan_to_num(x.to(device), nan=0.0, posinf=1.0, neginf=-1.0)
            y = y.to(device)
            logits = client_model(x)
            _, preds = torch.max(logits, 1)
            eval_samples += y.size(0)
            correct += (preds == y).sum().item()

    accuracy = correct / eval_samples * 100 if eval_samples > 0 else 0.0

    return client_model, avg_loss, trained_epochs, total_samples, accuracy

