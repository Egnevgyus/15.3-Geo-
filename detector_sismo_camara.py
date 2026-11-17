import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
if not ret:
  cap.release()
  exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
h_img, w_img = prev_gray.shape

feature_params = dict(
    maxCorners=200, qualityLevel=0.1, minDistance=10, blockSize=7
)
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
smoothed_mag = 0.0

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  magnitude = 0.0

  if p0 is not None and len(p0) > 10:
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None, **lk_params
    )

    if p1 is not None:
      good_old = p0[st == 1]
      good_new = p1[st == 1]

      if len(good_old) > 10:
        displacements = good_new - good_old
        magnitudes = np.linalg.norm(displacements, axis=1)
        magnitude = np.median(magnitudes)

        for pt in good_new:
          nx, ny = map(int, pt.ravel())
          cv2.circle(frame, (nx, ny), 3, (0, 255, 0), -1)

  smoothed_mag = (0.6 * smoothed_mag) + (0.4 * magnitude)

  if smoothed_mag < 1.5:
    smoothed_mag = 0.0

  estado = "Normal"
  color = (0, 255, 0)

  if smoothed_mag > 10.0:
    estado = "TERREMOTO - ALERTA ROJA"
    color = (0, 0, 255)
  elif smoothed_mag > 5.0:
    estado = "Sismo Fuerte - ALERTA NARANJA"
    color = (0, 140, 255)
  elif smoothed_mag > 0.0:
    estado = "Sismo Leve - ALERTA AMARILLA"
    color = (0, 255, 255)

  cv2.putText(
      frame,
      f"Velocidad Estructural: {smoothed_mag:.2f} px",
      (30, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      color,
      2,
  )
  cv2.putText(
      frame,
      f"Estado: {estado}",
      (30, 80),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      color,
      2,
  )

  if smoothed_mag > 10.0:
    cv2.rectangle(frame, (0, 0), (w_img - 1, h_img - 1), (0, 0, 255), 6)

  cv2.imshow("Monitor Sismico Estable", frame)

  prev_gray = curr_gray.copy()
  p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
  if p0 is None or len(p0) < 10:
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()