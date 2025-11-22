import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import EnhancerModel, load_diabetes_data


def load_flat_params_to_model(model, flat_params):
    """복호화된 flat 파라미터를 모델에 로드"""
    ptr = 0
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        numel = param.numel()
        arr = torch.from_numpy(flat_params[ptr:ptr + numel].astype(np.float32)).view(param.size())
        state_dict[name].copy_(arr)
        ptr += numel
    model.load_state_dict(state_dict)


def summarize_complication(prob, high=0.5, medium=0.3):
    if prob >= high:
        return "High (>=50%)"
    if prob >= medium:
        return "Moderate (30~50%)"
    return "Low (<30%)"


def main():
    device = torch.device('cpu')
    
    # 1. 데이터/모델 준비
    train_dataset, test_dataset = load_diabetes_data('diabetic_data.csv')
    input_dim = test_dataset.X.shape[1]
    num_classes = train_dataset.num_classes if hasattr(train_dataset, "num_classes") else 2
    model = EnhancerModel(input_dim=input_dim, num_classes=num_classes).to(device)
    
    # 복호화된 flat 파라미터 로드
    flat_params = np.load('decrypted_params.npy')
    load_flat_params_to_model(model, flat_params)
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 2. 예측 수행
    model.eval()
    diabetes_probs, diabetes_preds = [], []
    readmit_probs, readmit_preds = [], []
    complication_probs, complication_preds = [], []
    
    with torch.no_grad():
        for x, _, _, _, _ in test_loader:
            x = x.to(device)
            outputs = model(x, return_aux=True)
            if isinstance(outputs, tuple) and len(outputs) >= 4:
                main_logits, _, readmit_logits, complication_logits = outputs
            else:
                main_logits = outputs
                readmit_logits = complication_logits = None
            
            main_prob = torch.softmax(main_logits, dim=1)
            diabetes_probs.extend(main_prob[:, 1].cpu().numpy())
            diabetes_preds.extend(main_prob.argmax(dim=1).cpu().numpy())
            
            if readmit_logits is not None:
                r_prob = torch.softmax(readmit_logits, dim=1)
                readmit_probs.extend(r_prob[:, 1].cpu().numpy())
                readmit_preds.extend(r_prob.argmax(dim=1).cpu().numpy())
            else:
                readmit_probs.extend([0.0] * len(x))
                readmit_preds.extend([0] * len(x))
            
            if complication_logits is not None:
                c_prob = torch.softmax(complication_logits, dim=1)
                complication_probs.extend(c_prob[:, 1].cpu().numpy())
                complication_preds.extend(c_prob.argmax(dim=1).cpu().numpy())
            else:
                complication_probs.extend([0.0] * len(x))
                complication_preds.extend([0] * len(x))
    
    # 3. 원본 데이터 매칭 (train_test_split과 동일한 인덱스 사용)
    df_raw = pd.read_csv('diabetic_data.csv')
    df = df_raw.drop(columns=['encounter_id', 'patient_nbr'])
    df['readmitted'] = df['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
    _, test_indices = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=df['readmitted']
    )
    df_result = df_raw.iloc[test_indices].reset_index(drop=True)
    
    # 4. 한글 컬럼으로 결과 정리
    df_result['당뇨병_확률'] = diabetes_probs
    df_result['당뇨병_예측'] = diabetes_preds
    df_result['재입원_확률'] = readmit_probs
    df_result['재입원_예측'] = readmit_preds
    df_result['합병증_확률'] = complication_probs
    df_result['합병증_예측'] = complication_preds
    df_result['합병증_요약'] = df_result['합병증_확률'].apply(summarize_complication)
    
    output_path = 'prediction_results.xlsx'
    df_result.to_excel(output_path, index=False)
    print(f"엑셀 파일 저장 완료: {output_path} (행 수: {len(df_result)})")


if __name__ == "__main__":
    main()