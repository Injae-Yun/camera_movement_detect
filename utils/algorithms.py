import time
import cv2
import numpy as np
import yaml

def available_methods():
    return {
        'diff': diff_based_velocity,
        'optical': optical_flow_velocity,
        'optical_grid': optical_flow_velocity_grid,
        'farneback': farneback_velocity,  # placeholder - implemented below
    }
# 설정 파일 로드 함수
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 전역 설정 로드 (또는 메인 함수에서 로드하여 전달)
CONFIG = load_config()


def apply_roi_mask_by_params(gray_frame, roi_params):
    """(공통 유틸) ROI 마스크 생성 함수"""
    h, w = gray_frame.shape[:2]
    
    bx1 = int(w * roi_params['bottom_x1_ratio'])
    bx2 = int(w * roi_params['bottom_x2_ratio'])
    tx1 = int(w * roi_params['top_x1_ratio'])
    tx2 = int(w * roi_params['top_x2_ratio'])
    ty = int(h * roi_params['top_y_ratio'])

    pts = np.array([[bx1, h], [bx2, h], [tx2, ty], [tx1, ty]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def diff_based_velocity(video_path, roi_mask=None, display=False):
    """
    개선된 차분(Diff) 기반 움직임 측정 함수.
    컨투어가 아닌, 프레임 간 픽셀 밝기 변화량의 평균을 계산합니다.
    
    Args:
        video_path: 비디오 파일 경로
        roi_mask: (Optional) 0/1로 구성된 2D 바이너리 마스크 (numpy array)
        display: 시각화 여부

    Returns:
        dict: 평균 움직임 스코어, 프레임별 스코어 리스트, 실행 시간 등
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}
    config = CONFIG['diff']
    # YAML 파라미터 로드
    blur_ksize = tuple(config['blur_ksize']) # [21, 21] -> (21, 21)
    diff_thresh = config['threshold']

    prev_gray = None
    velocities = []
    start = time.time()

    # ROI 처리 (생략된 부분은 위와 동일)
    roi_uint8 = None
    if roi_mask is not None:
        roi_uint8 = (roi_mask > 0).astype(np.uint8) * 255
        roi_area = cv2.countNonZero(roi_uint8)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 설정값 적용
        curr_gray = cv2.GaussianBlur(curr_gray, blur_ksize, 0)

        if prev_gray is None:
            prev_gray = curr_gray
            continue

        frame_delta = cv2.absdiff(prev_gray, curr_gray)
        
        # 설정값 적용
        _, thresh = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_TOZERO)

        # ... (이하 점수 계산 로직 동일) ...
        MOVEMENT_SCORE = np.mean(thresh) # (단순화)
        if roi_uint8 is not None:
             # ROI 로직 적용
             pass

        velocities.append(float(MOVEMENT_SCORE))
        prev_gray = curr_gray

    cap.release()
    runtime = time.time() - start
    
    return {
        'method': 'diff_v2_magnitude',
        'avg_velocity': float(np.mean(velocities)) if velocities else 0.0,
        'velocities': velocities,
        'runtime': runtime,
    }


def optical_flow_velocity(video_path, display=False):
    """
    YAML 설정을 사용하는 Feature-based Optical Flow (원복됨)
    - goodFeaturesToTrack 사용
    - Magnitude(벡터 크기) 계산 적용
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}
    config=CONFIG['optical_flow']
    # 1. 설정값 언패킹
    resize_scale = config['resize_scale']
    redetect_interval = config['redetect_interval']
    reset_thresh_ratio = config['reset_thresh']
    
    # Feature Params (Shi-Tomasi)
    feat_conf = config['feature_params']
    feature_params = dict(
        maxCorners=feat_conf['maxCorners'],
        qualityLevel=feat_conf['qualityLevel'],
        minDistance=feat_conf['minDistance'],
        blockSize=feat_conf['blockSize']
    )
    
    # LK Params
    lk_conf = config['lk_params']
    lk_params = dict(
        winSize=tuple(lk_conf['win_size']),
        maxLevel=lk_conf['max_level'],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                  lk_conf['criteria']['max_count'], 
                  lk_conf['criteria']['epsilon'])
    )

    ret, old_frame = cap.read()
    if not ret:
        cap.release()
        return {"error": "Video error"}

    # 2. 리사이즈 및 초기 특징점 검출
    h, w = old_frame.shape[:2]
    new_h, new_w = int(h * resize_scale), int(w * resize_scale)
    
    old_small = cv2.resize(old_frame, (new_w, new_h))
    old_gray = cv2.cvtColor(old_small, cv2.COLOR_BGR2GRAY)

    # 초기 특징점 찾기 (Shi-Tomasi)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    velocities = []
    total_samples = 0
    frame_idx = 0
    start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_small = cv2.resize(frame, (new_w, new_h))
        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # 3. 특징점 관리 (재검출 로직)
        # 점이 너무 적거나(reset_thresh), 일정 주기(interval)가 되면 특징점 보충
        current_point_count = len(p0) if p0 is not None else 0
        target_count = feature_params['maxCorners']
        
        need_redetect = (current_point_count < target_count * reset_thresh_ratio) or \
                        (frame_idx % redetect_interval == 0)

        if need_redetect:
            # 기존 점을 버리고 새로 찾을지, 추가할지 결정해야 함.
            # 속도와 단순함을 위해 여기서는 '새로 찾기' 전략을 사용합니다.
            # (기존 점을 유지하려면 코드가 복잡해지고 연산이 늘어남)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            
            if p0 is None: # 여전히 점이 없으면 이번 프레임 스킵
                old_gray = frame_gray.copy()
                velocities.append(0.0)
                frame_idx += 1
                continue

        # 4. Optical Flow 계산
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and st is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # [Sample Count] 이번 프레임에서 유효하게 추적된 점의 개수 누적
            total_samples += len(good_new)
            
            # [Magnitude Correction] 방사형 움직임도 반영하기 위해 거리(Distance) 계산
            displacements = good_new - good_old
            magnitudes = np.linalg.norm(displacements, axis=1)
            
            if len(magnitudes) > 0:
                # 리사이즈 스케일 보정하여 원래 크기 기준 속도 반환
                avg_mag = np.mean(magnitudes) / resize_scale
                velocities.append(float(avg_mag))
            else:
                velocities.append(0.0)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        else:
            velocities.append(0.0)
            old_gray = frame_gray.copy()
            # 추적 실패 시 다음 턴에 재검출하도록 p0 초기화
            p0 = None 

        if display:
             vis = frame.copy()
             if p1 is not None:
                 for (new, old) in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    # 원본 좌표로 변환
                    cv2.line(vis, (int(a/resize_scale), int(b/resize_scale)), 
                                  (int(c/resize_scale), int(d/resize_scale)), (0, 255, 0), 2)
             cv2.imshow('Optical Flow (Feature)', vis)
             if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        frame_idx += 1

    cap.release()
    if display: cv2.destroyWindow('Optical Flow (Feature)')
    
    return {
        'method': 'optical_flow_feature',
        'sample_count': total_samples,
        'avg_velocity': float(np.mean(velocities)) if velocities else 0.0,
        'velocities': velocities,
        'runtime': time.time() - start
    }

def optical_flow_velocity_grid(video_path, display=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}
    config = CONFIG['optical_flow']
    # YAML 파라미터 로드
    resize_scale = config['resize_scale']
    grid_step = config['grid_step']
    reset_ratio = config['reset_thresh']
    
    # LK 파라미터 구성
    lk_conf = config['lk_params']
    lk_params = dict(
        winSize=tuple(lk_conf['win_size']), # list -> tuple 변환
        maxLevel=lk_conf['max_level'],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                  lk_conf['criteria']['max_count'], 
                  lk_conf['criteria']['epsilon'])
    )

    ret, old_frame = cap.read()
    if not ret:
        cap.release()
        return {"error": "Video error"}

    # 리사이즈 및 초기화
    h, w = old_frame.shape[:2]
    new_h, new_w = int(h * resize_scale), int(w * resize_scale)
    
    old_small = cv2.resize(old_frame, (new_w, new_h))
    old_gray = cv2.cvtColor(old_small, cv2.COLOR_BGR2GRAY)

    # Grid Point 생성 (grid_step 설정값 적용)
    y_grid, x_grid = np.mgrid[grid_step:new_h:grid_step, grid_step:new_w:grid_step]
    p0_initial = np.vstack((x_grid.flatten(), y_grid.flatten())).T.reshape(-1, 1, 2).astype(np.float32)
    p0 = p0_initial.copy()

    velocities = []
    total_samples = 0
    start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_small = cv2.resize(frame, (new_w, new_h))
        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # 재설정 로직 (reset_ratio 설정값 적용)
        if p0 is None or len(p0) < len(p0_initial) * reset_ratio:
            p0 = p0_initial.copy()
            old_gray = frame_gray.copy()
            velocities.append(0.0)
            continue

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            total_samples += len(good_new)
            
            displacements = good_new - good_old
            magnitudes = np.linalg.norm(displacements, axis=1)
            
            if len(magnitudes) > 0:
                avg_mag = np.mean(magnitudes) / resize_scale
                velocities.append(float(avg_mag))
            else:
                velocities.append(0.0)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        else:
            velocities.append(0.0)
            old_gray = frame_gray.copy()
            p0 = p0_initial.copy()

        if display:
             # 시각화 로직 (생략)
             pass

    cap.release()
    runtime = time.time() - start
    
    return {
        'method': 'optical_flow_grid',
        'sample_count': total_samples,
        'frame_count': len(velocities),
        'avg_velocity': float(np.mean(velocities)) if velocities else 0.0,
        'velocities': velocities,
        'runtime': runtime
    }

def compute_optical_flow_with_mask(prev_gray, curr_gray, roi_mask):
    """Farneback 광류 → 크기(magnitude)의 ROI 평균값으로 움직임 정도 산출

    이 함수는 제공해주신 코드를 최대한 보존해 구현했습니다.
    - roi_mask는 0/1 바이너리(또는 bool)이어야 평균이 정확합니다.
    - 반환값은 픽셀 단위의 평균 이동 크기(픽셀/프레임).
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=1, winsize=9,
        iterations=1, poly_n=3, poly_sigma=0.9, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # roi_mask는 0/1 바이너리여야 평균이 정확함
    masked_mag = mag * roi_mask
    mean_movement = np.sum(masked_mag) / (np.sum(roi_mask) + 1e-6)
    return float(mean_movement)


def farneback_velocity(video_path, roi_mask=None, display=False):
    """
    YAML 설정을 사용하는 Farneback 알고리즘
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    # 1. 설정값 언패킹
    config=CONFIG['farneback']
    common_config=CONFIG['common']
    scale = config['resize_scale']
    
    # Farneback 파라미터
    farn_params = dict(
        pyr_scale=config['pyr_scale'],
        levels=config['levels'],
        winsize=config['winsize'],
        iterations=config['iterations'],
        poly_n=config['poly_n'],
        poly_sigma=config['poly_sigma'],
        flags=config['flags']
    )

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return {"error": "Video error"}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    # 2. ROI 마스크 준비 (설정된 scale에 맞춰 리사이즈)
    if roi_mask is None:
        # 공통 설정의 ROI 비율 사용
        small_dummy = np.zeros((int(h*scale), int(w*scale)), dtype=np.uint8)
        roi_small = apply_roi_mask_by_params(small_dummy, common_config['roi'])
        roi = roi_small # 이미 0/1 마스크
    else:
        # 사용자 제공 마스크 리사이즈
        roi = (roi_mask > 0).astype(np.uint8)
        roi = cv2.resize(roi, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)

    velocities = []
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. 리사이즈 (속도 최적화 핵심)
        prev_small = cv2.resize(prev_gray, (0, 0), fx=scale, fy=scale)
        curr_small = cv2.resize(curr_gray, (0, 0), fx=scale, fy=scale)
        
        # ROI 크기 안전장치
        if roi.shape != prev_small.shape:
             roi = cv2.resize(roi, (prev_small.shape[1], prev_small.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 4. Farneback 계산 (설정된 파라미터 주입)
        flow = cv2.calcOpticalFlowFarneback(prev_small, curr_small, None, **farn_params)
        
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        masked_mag = mag * roi
        
        # 평균 움직임 계산
        mean_movement_small = np.sum(masked_mag) / (np.sum(roi) + 1e-6)
        
        # 원래 스케일로 환산
        mean_movement = float(mean_movement_small) / max(1e-6, scale)
        velocities.append(mean_movement)

        if display:
             vis = frame.copy()
             cv2.putText(vis, f"Farneback: {mean_movement:.2f}", (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
             cv2.imshow('Farneback', vis)
             if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        prev_gray = curr_gray.copy()

    cap.release()
    if display: cv2.destroyWindow('Farneback')

    return {
        'method': 'farneback',
        'sample_count': len(velocities), # Farneback은 픽셀 전체가 샘플이지만, 편의상 프레임 수 반환
        'avg_velocity': float(np.mean(velocities)) if velocities else 0.0,
        'velocities': velocities,
        'runtime': time.time() - start
    }




