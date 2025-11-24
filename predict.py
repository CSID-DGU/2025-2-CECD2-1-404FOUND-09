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
    
    # 데이터 레이블 분포 확인
    train_labels = train_dataset.y if hasattr(train_dataset, 'y') else None
    test_labels = test_dataset.y if hasattr(test_dataset, 'y') else None
    if train_labels is not None:
        train_diabetes_ratio = np.mean(train_labels == 1) * 100
        train_diabetes_count = np.sum(train_labels == 1)
        print(f"학습 데이터 당뇨병 비율: {train_diabetes_ratio:.1f}% ({train_diabetes_count}/{len(train_labels)})")
    if test_labels is not None:
        test_diabetes_ratio = np.mean(test_labels == 1) * 100
        test_diabetes_count = np.sum(test_labels == 1)
        print(f"테스트 데이터 당뇨병 비율: {test_diabetes_ratio:.1f}% ({test_diabetes_count}/{len(test_labels)})")
    
    # 모델 로드: 우선순위 1) trained_enhancer_model.pth, 2) global_model.pth, 3) decrypted_params.npy
    # trained_enhancer_model.pth를 우선 로드 (input_dim이 맞는 모델)
    model = None
    use_aux_outputs = False  # 보조 출력(합병증, 재입원) 사용 여부
    model_loaded = False
    
    if os.path.exists('trained_enhancer_model.pth'):
        print("trained_enhancer_model.pth 파일 로드 중...")
        try:
            checkpoint = torch.load('trained_enhancer_model.pth', map_location=device)
            saved_input_dim = checkpoint.get('input_dim', input_dim)
            
            # input_dim이 일치하는 경우에만 로드
            if saved_input_dim == input_dim:
                hidden_dims = checkpoint.get('hidden_dims', (128, 96, 64))
                if isinstance(hidden_dims, list):
                    hidden_dims = tuple(hidden_dims)
                dropout_rate = checkpoint.get('dropout_rate', 0.2)
                
                model = EnhancerModel(
                    input_dim=input_dim,
                    num_classes=checkpoint.get('num_classes', num_classes),
                    hidden_dims=hidden_dims,
                    dropout_rate=dropout_rate
                ).to(device)
                model.load_state_dict(checkpoint['state_dict'])
                
                # 모델 검증
                model.eval()
                with torch.no_grad():
                    test_input = torch.randn(1, input_dim).to(device)
                    test_output = model(test_input)
                    test_probs = torch.softmax(test_output, dim=1)
                    print(f"모델 테스트 확률: {test_probs.cpu().numpy()}")
                
                first_layer_weight = list(model.parameters())[0]
                weight_mean = first_layer_weight.mean().item()
                weight_std = first_layer_weight.std().item()
                print(f"모델 첫 레이어 가중치 통계: mean={weight_mean:.6f}, std={weight_std:.6f}")
                
                if abs(weight_mean) > 0.001 or weight_std > 0.1:
                    use_aux_outputs = True
                    model_loaded = True
                    print(f"trained_enhancer_model.pth 로드 완료: input_dim={input_dim}, num_classes={checkpoint.get('num_classes', num_classes)}")
                else:
                    print("경고: trained_enhancer_model.pth의 가중치가 초기화 상태입니다.")
                    model = None
            else:
                print(f"경고: trained_enhancer_model.pth의 input_dim({saved_input_dim})이 실제 데이터({input_dim})와 다릅니다.")
        except Exception as e:
            print(f"trained_enhancer_model.pth 로드 실패: {e}")
    
    # trained_enhancer_model.pth가 없거나 로드 실패한 경우 global_model.pth 시도
    if not model_loaded and os.path.exists('global_model.pth'):
        print("global_model.pth 파일 로드 중...")
        checkpoint = torch.load('global_model.pth', map_location=device, weights_only=False)
        model_type = checkpoint.get('model_type', 'Unknown')
        saved_input_dim = checkpoint.get('input_dim', None)
        saved_num_classes = checkpoint.get('num_classes', num_classes)
        
        # input_dim이 일치하는 경우에만 로드
        if saved_input_dim == input_dim:
            hidden_dims = checkpoint.get('hidden_dims', (128, 96, 64))
            if isinstance(hidden_dims, list):
                hidden_dims = tuple(hidden_dims)
            dropout_rate = checkpoint.get('dropout_rate', 0.2)
            
            model = EnhancerModel(
                input_dim=input_dim,
                num_classes=saved_num_classes,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            ).to(device)
            
            # state_dict 로드
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
                elif key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        skipped_layers.append(key)
            
            model.load_state_dict(filtered_state_dict, strict=False)
            
            # 첫 번째 레이어가 스킵되었는지 확인
            first_layer_skipped = any('input_projection.0.weight' in layer or 'input_projection.0' in layer for layer in skipped_layers)
            if first_layer_skipped:
                print(f"경고: 첫 번째 레이어(input_projection.0.weight)가 스킵되었습니다!")
                print(f"모델이 제대로 작동하지 않을 수 있습니다. trained_enhancer_model.pth를 사용하거나 모델을 다시 학습하세요.")
                model = None
            elif skipped_layers:
                print(f"경고: {len(skipped_layers)}개 레이어가 스킵되었습니다.")
                if len(skipped_layers) <= 5:
                    print(f"스킵된 레이어: {skipped_layers}")
                else:
                    print(f"스킵된 레이어 (처음 5개): {skipped_layers[:5]}...")
            
            if model is not None:
                # 모델 검증
                model.eval()
                with torch.no_grad():
                    test_input = torch.randn(1, input_dim).to(device)
                    test_output = model(test_input)
                    test_probs = torch.softmax(test_output, dim=1)
                    print(f"모델 테스트 출력 shape: {test_output.shape}")
                    print(f"모델 테스트 확률: {test_probs.cpu().numpy()}")
                
                first_layer_weight = list(model.parameters())[0]
                weight_mean = first_layer_weight.mean().item()
                weight_std = first_layer_weight.std().item()
                print(f"모델 첫 레이어 가중치 통계: mean={weight_mean:.6f}, std={weight_std:.6f}")
                
                if abs(weight_mean) > 0.001 or weight_std > 0.1:
                    use_aux_outputs = True
                    model_loaded = True
                    print(f"global_model.pth 로드 완료 (보조 출력 활성화): input_dim={input_dim}, num_classes={saved_num_classes}")
                else:
                    print("경고: 모델 가중치가 초기화된 상태입니다. 모델을 사용할 수 없습니다.")
                    model = None
        else:
            print(f"경고: global_model.pth의 input_dim({saved_input_dim})이 실제 데이터({input_dim})와 다릅니다.")
            print(f"모델을 로드할 수 없습니다. trained_enhancer_model.pth를 사용하거나 모델을 다시 학습하세요.")
    
    # trained_enhancer_model.pth와 global_model.pth 모두 실패한 경우 decrypted_params.npy 시도
    if not model_loaded and os.path.exists('decrypted_params.npy'):
        print("decrypted_params.npy 파일 로드 중...")
        try:
            model = EnhancerModel(input_dim=input_dim, num_classes=num_classes, hidden_dims=(128, 96, 64), dropout_rate=0.2).to(device)
            flat_params = np.load('decrypted_params.npy')
            load_flat_params_to_model(model, flat_params)
            
            # 모델 검증
            model.eval()
            with torch.no_grad():
                test_input = torch.randn(1, input_dim).to(device)
                test_output = model(test_input)
                test_probs = torch.softmax(test_output, dim=1)
                print(f"모델 테스트 확률: {test_probs.cpu().numpy()}")
            
            first_layer_weight = list(model.parameters())[0]
            weight_mean = first_layer_weight.mean().item()
            weight_std = first_layer_weight.std().item()
            print(f"모델 첫 레이어 가중치 통계: mean={weight_mean:.6f}, std={weight_std:.6f}")
            
            if abs(weight_mean) > 0.001 or weight_std > 0.1:
                use_aux_outputs = True
                model_loaded = True
                print(f"파라미터 로드 완료: {len(flat_params):,} 파라미터")
            else:
                print("경고: 모델 가중치가 초기화된 상태입니다.")
                model = None
        except Exception as e:
            print(f"decrypted_params.npy 로드 실패: {e}")
            model = None
    
    # 모든 모델 로드 실패
    if not model_loaded or model is None:
        print("\n경고: 학습된 모델을 로드할 수 없습니다!")
        print("가능한 원인:")
        print("  1. 모델 파일이 없거나 손상되었습니다.")
        print("  2. 모델의 input_dim이 실제 데이터와 일치하지 않습니다.")
        print("  3. 모델이 제대로 학습되지 않았습니다.")
        print("\n해결 방법:")
        print("  1. FedHBClient.py를 실행하여 모델을 학습하세요.")
        print("  2. trained_enhancer_model.pth 파일이 있는지 확인하세요.")
        print("  3. 모델의 input_dim이 실제 데이터와 일치하는지 확인하세요.")
        return
    
    # 모델이 제대로 로드되었는지 최종 확인
    model.eval()
    first_layer_weight = list(model.parameters())[0]
    weight_mean = first_layer_weight.mean().item()
    weight_std = first_layer_weight.std().item()
    
    if abs(weight_mean) < 0.001 and weight_std < 0.1:
        print("\n경고: 모델 가중치가 초기화된 상태입니다. 예측이 제대로 작동하지 않을 수 있습니다.")
        print("모델을 다시 학습하거나 올바른 모델 파일을 사용하세요.")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 2. 예측 수행
    model.eval()
    diabetes_probs, diabetes_preds = [], []
    readmit_probs, readmit_preds = [], []
    complication_probs, complication_preds = [], []
    
    with torch.no_grad():
        first = True
        for batch_idx, batch in enumerate(test_loader):
            # 데이터셋이 5-tuple 또는 6-tuple 반환 가능 (sample_weights 포함)
            if len(batch) == 6:
                x, _, _, _, _, _ = batch
            elif len(batch) == 5:
                x, _, _, _, _ = batch
            else:
                x = batch[0]
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
                        pass  # warning 메시지 제거
                    else:
                        pass
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
                print(f"  main_logits 샘플 (첫 5개): {main_logits[:5].cpu().numpy()}")
                # logits의 다양성 확인
                logits_np = main_logits.cpu().numpy()
                print(f"  logits 통계:")
                print(f"    - min: {logits_np.min():.6f}, max: {logits_np.max():.6f}")
                print(f"    - mean: {logits_np.mean():.6f}, std: {logits_np.std():.6f}")
                unique_logits_4 = len(np.unique(np.round(logits_np, 4)))
                unique_logits_6 = len(np.unique(np.round(logits_np, 6)))
                print(f"    - 유니크 logits 개수 (소수 4자리): {unique_logits_4}")
                print(f"    - 유니크 logits 개수 (소수 6자리): {unique_logits_6}")
                
                # 모든 logits가 동일한지 확인
                if unique_logits_4 == 1:
                    print(f"\n    ⚠️ 경고: 모든 logits가 동일합니다! 모델이 모든 입력에 대해 동일한 출력을 내고 있습니다.")
                    print(f"    logits 값: {logits_np[0]}")
                    print(f"    원인 분석:")
                    print(f"      1. 모델의 첫 번째 레이어(input_projection)가 스킵되어 입력이 무시되고 있습니다.")
                    print(f"      2. 모델이 제대로 학습되지 않았거나 초기화 상태입니다.")
                    print(f"      3. 모델의 가중치가 모두 동일하거나 거의 동일합니다.")
                elif unique_logits_6 == 1:
                    print(f"    경고: 모든 logits가 거의 동일합니다!")
                
                # 첫 5개 샘플의 logits 차이 확인
                if len(main_logits) >= 5:
                    for i in range(4):
                        diff = (main_logits[i] - main_logits[i+1]).abs().sum().item()
                        print(f"    - 샘플 {i}와 {i+1}의 logits 차이 합: {diff:.6f}")
                        if diff < 1e-6:
                            print(f"      경고: 샘플 {i}와 {i+1}의 logits가 거의 동일합니다!")
                
                # 입력 데이터 다양성도 확인
                print(f"\n  입력 데이터 다양성 확인:")
                x_np = x.cpu().numpy()
                print(f"    - 입력 shape: {x_np.shape}")
                print(f"    - 입력 min: {x_np.min():.6f}, max: {x_np.max():.6f}")
                print(f"    - 입력 mean: {x_np.mean():.6f}, std: {x_np.std():.6f}")
                if len(x) >= 5:
                    for i in range(4):
                        input_diff = (x[i] - x[i+1]).abs().sum().item()
                        print(f"    - 샘플 {i}와 {i+1}의 입력 차이 합: {input_diff:.6f}")
                        if input_diff < 1e-6:
                            print(f"      경고: 샘플 {i}와 {i+1}의 입력이 거의 동일합니다!")
                
                # 모델의 중간 출력 확인 (첫 번째 레이어 출력)
                if hasattr(model, 'input_projection'):
                    with torch.no_grad():
                        model.eval()
                        first_output = model.input_projection(x[:3])
                        print(f"\n  모델 중간 출력 확인 (input_projection 출력, 첫 3개 샘플):")
                        print(f"    - shape: {first_output.shape}")
                        print(f"    - 샘플 0: {first_output[0][:5].cpu().numpy()} ...")
                        print(f"    - 샘플 1: {first_output[1][:5].cpu().numpy()} ...")
                        print(f"    - 샘플 2: {first_output[2][:5].cpu().numpy()} ...")
                        # 첫 번째 레이어 출력이 모두 동일한지 확인
                        diff_01 = (first_output[0] - first_output[1]).abs().sum().item()
                        diff_02 = (first_output[0] - first_output[2]).abs().sum().item()
                        print(f"    - 샘플 간 차이: 0-1={diff_01:.6f}, 0-2={diff_02:.6f}")
                        if diff_01 < 1e-6 or diff_02 < 1e-6:
                            print(f"    ⚠️ 경고: input_projection 출력이 동일합니다! 첫 번째 레이어가 제대로 작동하지 않습니다.")
            
            # 모델 출력 안정화: temperature scaling 적용
            temperature = 1.0  # 기본값, 필요시 조정 가능
            main_prob = torch.softmax(main_logits / temperature, dim=1)
            batch_diabetes_probs = main_prob[:, 1].cpu().numpy()
            
            # 확률 값 검증 및 안정화
            batch_diabetes_probs = np.clip(batch_diabetes_probs, 0.0, 1.0)
            # NaN/Inf 처리
            batch_diabetes_probs = np.nan_to_num(batch_diabetes_probs, nan=0.5, posinf=1.0, neginf=0.0)
            
            diabetes_probs.extend(batch_diabetes_probs)
            diabetes_preds.extend(main_prob.argmax(dim=1).cpu().numpy())
            
            # === 디버깅: 확률 확인 (첫 배치만) ===
            if batch_idx == 0:
                print(f"\n  === 확률 확인 (첫 배치) ===")
                print(f"  당뇨병 확률 (첫 10개): {batch_diabetes_probs[:10]}")
                print(f"  확률 min: {batch_diabetes_probs.min():.6f}, max: {batch_diabetes_probs.max():.6f}")
                print(f"  확률 mean: {batch_diabetes_probs.mean():.6f}, std: {batch_diabetes_probs.std():.6f}")
                # 확률이 모두 동일한지 확인
                unique_probs = np.unique(np.round(batch_diabetes_probs, 6))
                print(f"  유니크 확률 개수 (소수 6자리): {len(unique_probs)}")
                if len(unique_probs) == 1:
                    print(f"  경고: 모든 확률이 동일합니다! ({unique_probs[0]:.6f})")
                    print(f"  원인 분석:")
                    print(f"    - 모델이 모든 입력에 대해 동일한 logits를 출력하고 있습니다.")
                    print(f"    - 모델이 제대로 학습되지 않았거나 초기화 상태일 수 있습니다.")
                    print(f"    - 모델의 첫 번째 레이어가 스킵되어 입력이 제대로 처리되지 않을 수 있습니다.")
                elif len(unique_probs) < 5:
                    print(f"  경고: 확률 다양성이 낮습니다. 유니크 확률: {unique_probs[:10]}")
            
            if readmit_logits is not None:
                r_prob = torch.softmax(readmit_logits, dim=1)
                readmit_probs.extend(r_prob[:, 1].cpu().numpy())
                readmit_preds.extend(r_prob.argmax(dim=1).cpu().numpy())
            else:
                readmit_probs.extend([0.0] * len(x))
                readmit_preds.extend([0] * len(x))
            
            if complication_logits is not None:
                c_prob = torch.softmax(complication_logits, dim=1)
                # 합병증 확률은 당뇨병 확률에 조건부로 계산
                # 합병증 확률 = 당뇨병 확률 × 합병증 조건부 확률
                raw_complication_probs = c_prob[:, 1].cpu().numpy()
                # 합병증 확률을 당뇨병 확률과 곱하여 조건부 확률로 계산
                # 이렇게 하면 당뇨병 확률이 낮으면 합병증 확률도 자연스럽게 낮아짐
                adjusted_complication_probs = batch_diabetes_probs * raw_complication_probs
                # 합병증 확률이 당뇨병 확률을 초과하지 않도록 제한
                adjusted_complication_probs = np.minimum(adjusted_complication_probs, batch_diabetes_probs)
                # 0과 1 사이로 클리핑
                adjusted_complication_probs = np.clip(adjusted_complication_probs, 0.0, 1.0)
                complication_probs.extend(adjusted_complication_probs)
                complication_preds.extend((adjusted_complication_probs >= 0.5).astype(int))
            else:
                complication_probs.extend([0.0] * len(x))
                complication_preds.extend([0] * len(x))
    
    # === 디버깅: 전체 확률 배열 통계 ===
    print(f"\n=== [디버깅] 전체 예측 결과 통계 ===")
    diabetes_probs_arr = np.array(diabetes_probs)
    print(f"당뇨병 확률:")
    print(f"  - 총 샘플 수: {len(diabetes_probs_arr)}")
    unique_4 = len(np.unique(np.round(diabetes_probs_arr, 4)))
    unique_6 = len(np.unique(np.round(diabetes_probs_arr, 6)))
    print(f"  - 유니크 확률 개수 (소수 4자리까지): {unique_4}")
    print(f"  - 유니크 확률 개수 (소수 6자리까지): {unique_6}")
    print(f"  - min: {diabetes_probs_arr.min():.6f}, max: {diabetes_probs_arr.max():.6f}")
    print(f"  - mean: {diabetes_probs_arr.mean():.6f}, std: {diabetes_probs_arr.std():.6f}")
    print(f"  - 중앙값: {np.median(diabetes_probs_arr):.6f}")
    print(f"  - 25% 분위수: {np.percentile(diabetes_probs_arr, 25):.6f}")
    print(f"  - 75% 분위수: {np.percentile(diabetes_probs_arr, 75):.6f}")
    
    # 확률 분포 확인
    high_prob_count = np.sum(diabetes_probs_arr >= 0.7)
    medium_prob_count = np.sum((diabetes_probs_arr >= 0.4) & (diabetes_probs_arr < 0.7))
    low_prob_count = np.sum(diabetes_probs_arr < 0.4)
    print(f"  - 확률 분포: 높음(>=70%): {high_prob_count}개 ({high_prob_count/len(diabetes_probs_arr)*100:.1f}%), "
          f"중간(40-70%): {medium_prob_count}개 ({medium_prob_count/len(diabetes_probs_arr)*100:.1f}%), "
          f"낮음(<40%): {low_prob_count}개 ({low_prob_count/len(diabetes_probs_arr)*100:.1f}%)")
    
    # 모든 확률이 동일한지 확인
    if unique_4 == 1:
        print(f"\n경고: 모든 확률이 동일합니다! (소수 4자리까지: {np.unique(np.round(diabetes_probs_arr, 4))[0]:.4f})")
        print("가능한 원인:")
        print("  1. 모델이 제대로 학습되지 않았을 수 있습니다.")
        print("  2. 모델의 가중치가 초기화 상태일 수 있습니다.")
        print("  3. 모든 입력 데이터가 동일하게 전처리되었을 수 있습니다.")
        print("  4. 모델이 모든 입력에 대해 동일한 출력을 내고 있습니다.")
    elif unique_6 == 1:
        print(f"\n경고: 모든 확률이 거의 동일합니다! (소수 6자리까지: {np.unique(np.round(diabetes_probs_arr, 6))[0]:.6f})")
    elif unique_4 < 10:
        print(f"\n경고: 확률 다양성이 매우 낮습니다. 유니크 확률 개수: {unique_4}")
        print(f"유니크 확률 값들 (처음 20개): {np.unique(np.round(diabetes_probs_arr, 4))[:20]}")
    
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
    
    # diabetes_flag 생성 (예측에 사용한 것과 동일한 방식)
    from model import _ensure_diabetes_label
    df = _ensure_diabetes_label(df, 'diabetes_flag')
    
    # 실제 레이블 분포 확인
    actual_diabetes_ratio = np.mean(df['diabetes_flag'] == 1) * 100
    print(f"\n원본 데이터 당뇨병 비율: {actual_diabetes_ratio:.1f}%")
    
    _, test_indices = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=df['readmitted']
    )
    df_result = df_raw.iloc[test_indices].reset_index(drop=True)
    
    # 테스트 세트의 실제 당뇨병 레이블 확인
    test_actual_labels = df.iloc[test_indices]['diabetes_flag'].values
    test_actual_ratio = np.mean(test_actual_labels == 1) * 100
    print(f"테스트 세트 실제 당뇨병 비율: {test_actual_ratio:.1f}%")
    
    # 예측과 실제 레이블 비교
    if len(diabetes_probs) == len(test_actual_labels):
        # 높은 확률(>=0.7)로 예측한 것 중 실제 당뇨병인 비율
        high_prob_mask = np.array(diabetes_probs) >= 0.7
        if np.sum(high_prob_mask) > 0:
            high_prob_accuracy = np.mean(test_actual_labels[high_prob_mask] == 1) * 100
            print(f"높은 확률(>=70%) 예측 중 실제 당뇨병 비율: {high_prob_accuracy:.1f}%")
        # 실제 당뇨병인 환자들의 평균 예측 확률
        diabetes_patients_mask = test_actual_labels == 1
        if np.sum(diabetes_patients_mask) > 0:
            diabetes_patients_avg_prob = np.mean(np.array(diabetes_probs)[diabetes_patients_mask]) * 100
            print(f"실제 당뇨병 환자들의 평균 예측 확률: {diabetes_patients_avg_prob:.1f}%")
    
    # 4. 필요한 컬럼만 선택하여 결과 정리
    # 원하는 컬럼: encounter_id, patient_nbr, race, gender, age, 당뇨병_확률, 합병증_확률
    selected_columns = []
    
    # 기본 컬럼들 확인 및 추가
    if 'encounter_id' in df_result.columns:
        selected_columns.append('encounter_id')
    if 'patient_nbr' in df_result.columns:
        selected_columns.append('patient_nbr')
    if 'race' in df_result.columns:
        selected_columns.append('race')
    if 'gender' in df_result.columns:
        selected_columns.append('gender')
    if 'age' in df_result.columns:
        selected_columns.append('age')
    
    # 예측 결과 추가
    df_result['당뇨병_확률'] = diabetes_probs
    df_result['합병증_확률'] = complication_probs
    
    # 최종 컬럼 선택
    selected_columns.extend(['당뇨병_확률', '합병증_확률'])
    
    # 존재하는 컬럼만 선택
    available_columns = [col for col in selected_columns if col in df_result.columns]
    df_result = df_result[available_columns]
    
    # === 디버깅: 엑셀 저장 전 DataFrame 확인 ===
    print(f"\n=== [디버깅] 엑셀 저장 전 DataFrame 통계 ===")
    print(f"DataFrame shape: {df_result.shape}")
    print(f"선택된 컬럼: {list(df_result.columns)}")
    print(f"\n확률 컬럼 상위 10개:")
    print(df_result[['당뇨병_확률', '합병증_확률']].head(10))
    print(f"\n확률 컬럼 통계:")
    print(df_result[['당뇨병_확률', '합병증_확률']].describe())
    
    # 유니크 값 개수 확인
    unique_diabetes = df_result['당뇨병_확률'].nunique()
    unique_complication = df_result['합병증_확률'].nunique()
    print(f"\n유니크 값 개수:")
    print(f"  - 당뇨병_확률: {unique_diabetes}개")
    print(f"  - 합병증_확률: {unique_complication}개")
    
    if unique_diabetes == 1:
        pass  # warning 메시지 제거
    if unique_complication == 1:
        pass  # warning 메시지 제거
    
    output_path = 'prediction_results.xlsx'
    print(f"\n엑셀 파일 저장 시작: {output_path} (행 수: {len(df_result)}, 열 수: {len(df_result.columns)})")
    try:
        # openpyxl 엔진을 명시적으로 사용
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False, sheet_name='예측결과')
        print(f"엑셀 파일 저장 완료: {output_path}")
    except Exception as e:
        print(f"엑셀 저장 실패: {e}")
        print("CSV 파일로 대체 저장 중...")
        csv_path = output_path.replace('.xlsx', '.csv')
        df_result.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV 파일 저장 완료: {csv_path}")


if __name__ == "__main__":
    main()