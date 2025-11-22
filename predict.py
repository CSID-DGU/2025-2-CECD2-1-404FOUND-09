import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import EnhancerModel, load_diabetes_data
from improved_model import ImprovedEnhancerModel


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
    
    # 모델 로드: 우선순위 1) global_model.pth, 2) trained_enhancer_model.pth, 3) decrypted_params.npy
    model = None
    use_aux_outputs = False  # 보조 출력(합병증, 재입원) 사용 여부
    
    if os.path.exists('global_model.pth'):
        print("global_model.pth 파일 로드 중...")
        checkpoint = torch.load('global_model.pth', map_location=device, weights_only=False)
        model_type = checkpoint.get('model_type', 'Unknown')
        # 실제 데이터의 input_dim 사용 (저장된 값이 아닌)
        saved_num_classes = checkpoint.get('num_classes', num_classes)
        
        # 합병증/재입원 예측을 위해 항상 EnhancerModel 사용 (보조 출력 있음)
        # ImprovedEnhancerModel로 저장되어 있어도 EnhancerModel로 로드 시도
        hidden_dims = checkpoint.get('hidden_dims', (128, 96, 64))
        if isinstance(hidden_dims, list):
            hidden_dims = tuple(hidden_dims)
        dropout_rate = checkpoint.get('dropout_rate', 0.2)
        
        model = EnhancerModel(
            input_dim=input_dim,  # 실제 데이터의 input_dim 사용
            num_classes=saved_num_classes,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(device)
        
        # state_dict 로드 시 input_dim이 다른 경우 첫 번째 레이어만 스킵
        state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        skipped_layers = []
        
        for key, value in state_dict.items():
            # ImprovedEnhancerModel의 'feature_extractor'를 EnhancerModel의 'input_projection'으로 매핑
            mapped_key = key
            if 'feature_extractor' in key:
                mapped_key = key.replace('feature_extractor', 'input_projection')
            
            if mapped_key in model_state_dict:
                if model_state_dict[mapped_key].shape == value.shape:
                    filtered_state_dict[mapped_key] = value
                else:
                    skipped_layers.append(mapped_key)
                    print(f"⚠️  레이어 '{mapped_key}' 크기 불일치: 모델={model_state_dict[mapped_key].shape}, 저장된={value.shape} - 스킵")
            elif key in model_state_dict:
                # 원래 키 이름으로도 시도
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    skipped_layers.append(key)
                    print(f"⚠️  레이어 '{key}' 크기 불일치: 모델={model_state_dict[key].shape}, 저장된={value.shape} - 스킵")
            else:
                # ImprovedEnhancerModel에만 있는 레이어는 스킵
                if 'feature_extractor' in key or 'classifier' in key:
                    print(f"⚠️  레이어 '{key}'가 EnhancerModel에 없음 - 스킵 (ImprovedEnhancerModel 전용 레이어)")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        if skipped_layers:
            print(f"⚠️  주의: {len(skipped_layers)}개 레이어가 초기화된 상태입니다. (input_dim 불일치: 저장된={checkpoint.get('input_dim', 'Unknown')}, 실제={input_dim})")
            print(f"   스킵된 레이어: {', '.join(skipped_layers)}")
            print(f"   예측은 가능하지만 성능이 떨어질 수 있습니다.")
        
        use_aux_outputs = True
        print(f"EnhancerModel 로드 완료 (보조 출력 활성화): input_dim={input_dim} (실제 데이터), num_classes={saved_num_classes}")
    elif os.path.exists('trained_enhancer_model.pth'):
        print("trained_enhancer_model.pth 파일 로드 중...")
        checkpoint = torch.load('trained_enhancer_model.pth', map_location=device)
        hidden_dims = checkpoint.get('hidden_dims', (128, 96, 64))
        dropout_rate = checkpoint.get('dropout_rate', 0.2)
        model = EnhancerModel(
            input_dim=checkpoint.get('input_dim', input_dim),
            num_classes=checkpoint.get('num_classes', num_classes),
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        use_aux_outputs = True
        print(f"EnhancerModel 로드 완료: input_dim={checkpoint.get('input_dim', input_dim)}, num_classes={checkpoint.get('num_classes', num_classes)}")
    elif os.path.exists('decrypted_params.npy'):
        print("decrypted_params.npy 파일 로드 중...")
        model = EnhancerModel(input_dim=input_dim, num_classes=num_classes, hidden_dims=(128, 96, 64), dropout_rate=0.2).to(device)
        flat_params = np.load('decrypted_params.npy')
        load_flat_params_to_model(model, flat_params)
        use_aux_outputs = True
        print(f"파라미터 로드 완료: {len(flat_params):,} 파라미터")
    else:
        print("⚠️  저장된 모델 파일이 없습니다. 초기화된 모델을 사용합니다.")
        print("   학습을 먼저 실행하거나 모델 파일을 준비해주세요.")
        model = EnhancerModel(input_dim=input_dim, num_classes=num_classes, hidden_dims=(128, 96, 64), dropout_rate=0.2).to(device)
        use_aux_outputs = True
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 2. 예측 수행
    model.eval()
    diabetes_probs, diabetes_preds = [], []
    readmit_probs, readmit_preds = [], []
    complication_probs, complication_preds = [], []
    
    with torch.no_grad():
        first = True
        for batch_idx, (x, _, _, _, _) in enumerate(test_loader):
            x = x.to(device)
            
            # === 디버깅: 입력 데이터 확인 (첫 배치만) ===
            if first:
                print("\n=== [디버깅] 입력 첫 배치 x[:3] ===")
                print(f"첫 3개 샘플의 입력 벡터:")
                for i in range(min(3, len(x))):
                    print(f"  샘플 {i}: {x[i][:10].cpu().numpy()} ... (총 {x.shape[1]}차원)")
                print(f"입력 데이터 통계:")
                print(f"  - shape: {x.shape}")
                print(f"  - min: {x.min().item():.6f}, max: {x.max().item():.6f}")
                print(f"  - mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
                # 첫 3개 샘플이 서로 다른지 확인
                if len(x) >= 3:
                    diff_01 = (x[0] - x[1]).abs().sum().item()
                    diff_02 = (x[0] - x[2]).abs().sum().item()
                    diff_12 = (x[1] - x[2]).abs().sum().item()
                    print(f"  - 샘플 간 차이 합: 0-1={diff_01:.6f}, 0-2={diff_02:.6f}, 1-2={diff_12:.6f}")
                    if diff_01 < 1e-6 or diff_02 < 1e-6 or diff_12 < 1e-6:
                        print("  ⚠️ 경고: 입력 샘플들이 거의 동일합니다!")
                    else:
                        print("  ✓ 입력 샘플들이 서로 다릅니다.")
                first = False
            
            if use_aux_outputs:
                # EnhancerModel: 보조 출력 사용
                try:
                    outputs = model(x, return_aux=True)
                    if isinstance(outputs, tuple) and len(outputs) >= 4:
                        main_logits, _, readmit_logits, complication_logits = outputs
                    else:
                        main_logits = outputs
                        readmit_logits = complication_logits = None
                except TypeError:
                    # return_aux 파라미터가 없는 경우 (ImprovedEnhancerModel)
                    main_logits = model(x)
                    readmit_logits = complication_logits = None
            else:
                # ImprovedEnhancerModel: 보조 출력 없음
                main_logits = model(x)
                readmit_logits = complication_logits = None
            
            # === 디버깅: 모델 출력 확인 (첫 배치만) ===
            if batch_idx == 0:
                print(f"\n=== [디버깅] 모델 출력 (첫 배치) ===")
                print(f"  main_logits shape: {main_logits.shape}")
                print(f"  main_logits 샘플 (첫 3개): {main_logits[:3].cpu().numpy()}")
            
            main_prob = torch.softmax(main_logits, dim=1)
            batch_diabetes_probs = main_prob[:, 1].cpu().numpy()
            diabetes_probs.extend(batch_diabetes_probs)
            diabetes_preds.extend(main_prob.argmax(dim=1).cpu().numpy())
            
            # === 디버깅: 확률 확인 (첫 배치만) ===
            if batch_idx == 0:
                print(f"  당뇨병 확률 (첫 10개): {batch_diabetes_probs[:10]}")
                print(f"  확률 min: {batch_diabetes_probs.min():.6f}, max: {batch_diabetes_probs.max():.6f}")
                print(f"  확률 mean: {batch_diabetes_probs.mean():.6f}, std: {batch_diabetes_probs.std():.6f}")
            
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
    
    # === 디버깅: 전체 확률 배열 통계 ===
    print(f"\n=== [디버깅] 전체 예측 결과 통계 ===")
    diabetes_probs_arr = np.array(diabetes_probs)
    print(f"당뇨병 확률:")
    print(f"  - 총 샘플 수: {len(diabetes_probs_arr)}")
    print(f"  - 유니크 확률 개수 (소수 4자리까지): {len(np.unique(np.round(diabetes_probs_arr, 4)))}")
    print(f"  - 유니크 확률 개수 (소수 6자리까지): {len(np.unique(np.round(diabetes_probs_arr, 6)))}")
    print(f"  - min: {diabetes_probs_arr.min():.6f}, max: {diabetes_probs_arr.max():.6f}")
    print(f"  - mean: {diabetes_probs_arr.mean():.6f}, std: {diabetes_probs_arr.std():.6f}")
    if len(np.unique(np.round(diabetes_probs_arr, 4))) == 1:
        print("  ⚠️ 경고: 모든 샘플의 확률이 동일합니다 (소수 4자리까지)!")
    elif len(np.unique(np.round(diabetes_probs_arr, 6))) == 1:
        print("  ⚠️ 경고: 모든 샘플의 확률이 동일합니다 (소수 6자리까지)!")
    else:
        print("  ✓ 확률들이 서로 다릅니다.")
    
    if len(complication_probs) > 0 and not all(p == 0.0 for p in complication_probs):
        complication_probs_arr = np.array(complication_probs)
        print(f"\n합병증 확률:")
        print(f"  - 유니크 확률 개수 (소수 4자리까지): {len(np.unique(np.round(complication_probs_arr, 4)))}")
        print(f"  - min: {complication_probs_arr.min():.6f}, max: {complication_probs_arr.max():.6f}")
        print(f"  - mean: {complication_probs_arr.mean():.6f}, std: {complication_probs_arr.std():.6f}")
    
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
    
    # === 디버깅: 엑셀 저장 전 DataFrame 확인 ===
    print(f"\n=== [디버깅] 엑셀 저장 전 DataFrame 통계 ===")
    print(f"DataFrame shape: {df_result.shape}")
    print(f"\n확률 컬럼 상위 10개:")
    print(df_result[['당뇨병_확률', '재입원_확률', '합병증_확률']].head(10))
    print(f"\n확률 컬럼 통계:")
    print(df_result[['당뇨병_확률', '재입원_확률', '합병증_확률']].describe())
    
    # 유니크 값 개수 확인
    unique_diabetes = df_result['당뇨병_확률'].nunique()
    unique_readmit = df_result['재입원_확률'].nunique()
    unique_complication = df_result['합병증_확률'].nunique()
    print(f"\n유니크 값 개수:")
    print(f"  - 당뇨병_확률: {unique_diabetes}개")
    print(f"  - 재입원_확률: {unique_readmit}개")
    print(f"  - 합병증_확률: {unique_complication}개")
    
    if unique_diabetes == 1:
        print("  ⚠️ 경고: 당뇨병_확률이 모든 행에서 동일합니다!")
    if unique_readmit == 1:
        print("  ⚠️ 경고: 재입원_확률이 모든 행에서 동일합니다!")
    if unique_complication == 1:
        print("  ⚠️ 경고: 합병증_확률이 모든 행에서 동일합니다!")
    
    output_path = 'prediction_results.xlsx'
    print(f"\n엑셀 파일 저장 시작: {output_path} (행 수: {len(df_result)}, 열 수: {len(df_result.columns)})")
    try:
        # openpyxl 엔진을 명시적으로 사용
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False, sheet_name='예측결과')
        print(f"✅ 엑셀 파일 저장 완료: {output_path}")
    except Exception as e:
        print(f"⚠️  엑셀 저장 실패: {e}")
        print("CSV 파일로 대체 저장 중...")
        csv_path = output_path.replace('.xlsx', '.csv')
        df_result.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 파일 저장 완료: {csv_path}")


if __name__ == "__main__":
    main()