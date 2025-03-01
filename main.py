from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
import math


def normalize_points(points : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate centroid
    x_mean = np.mean(points[0,:])
    y_mean = np.mean(points[1,:])

    # Calculate the average distance of the points having the centroid as origin
    dist = np.sqrt((points[0,:] - x_mean)**2 + (points[1,:] - y_mean)**2)
    avg_dist = np.mean(dist)

    # Define the scale to have the average distance as sqrt(2)
    if avg_dist != 0:
        scale = np.sqrt(2)/avg_dist
    else:
        scale = 1

    # points = np.hstack((points,np.ones((points.shape[0],1)))).T
    # Define the normalization matrix (similar transformation)
    T = np.array([[scale,0,-scale*x_mean],
                  [0,scale,-scale*y_mean],
                  [0,0,1]])

    # Normalize points
    norm_pts = np.dot(T,points)

    return T, norm_pts

def compute_dlt_from_pairs(pairs : np.ndarray) -> np.ndarray:
    A = []
    for x1, y1, x2, y2 in pairs:
         A.append(np.array([
            [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2],
            [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1,-x2],
            [-y2 * x1, -y2 * y1, -y2, x2 * x1, x2 * y1, x2, 0, 0, 0]
        ]))
         
    A = np.array(A).reshape(-1, 9)
    _,_,Vt = np.linalg.svd(A)
    H = Vt[-1,:].reshape(3,3)
    return H

def compute_normalized_dlt(point_map : np.ndarray) -> np.ndarray:

    # Separate points
    pts1 = np.array([[point_map[i][0][0], point_map[i][0][1]] for i in range(len(point_map))])
    pts2 = np.array([[point_map[i][0][2], point_map[i][0][3]] for i in range(len(point_map))])

    # Add the homogeneous coordinate
    pts1 = np.hstack((pts1,np.ones((pts1.shape[0],1)))).T
    pts2 = np.hstack((pts2,np.ones((pts2.shape[0],1)))).T

    # Normalize points
    T1, norm_pts1 = normalize_points(pts1)
    T2, norm_pts2 = normalize_points(pts2)

    norm_pairs = np.array([(norm_pts1[0,i], norm_pts1[1,i], norm_pts2[0,i], norm_pts2[1,i]) for i in range(norm_pts1.shape[1])])

    # Compute DLT
    H_matrix = compute_dlt_from_pairs(norm_pairs)

    # Perform DLT and obtain normalized matrix
    H_matrix = np.dot(np.linalg.inv(T2),np.dot(H_matrix,T1))

    return H_matrix

def distance(point_map : np.ndarray,  H : np.ndarray) -> float:

    # Separate points
    pt1 = point_map[:2]
    pt2 = point_map[2:]

    # Calculate the distance between the points
    pt1 = np.array([pt1[0],pt1[1],1]).T
    pt2 = np.array([pt2[0],pt2[1],1]).T

    # Calculate the distance
    pt2_ = np.dot(H, pt1)
    if pt2_[2] != 0:
        pt2_ = pt2_/pt2_[2]

    return np.linalg.norm(pt2 - pt2_)

def ransac(point_map : np.ndarray, THRESHOLD : float = 2.5, MAX_ITERATIONS : int = 10000, MIN_INLIERS : float = 0.8) -> np.ndarray:
    
    print(f'RANSAC Homography Estimation with {len(point_map)} points')
    
    start = time.time()
    number_of_points = len(point_map)
    best_inliers : set = set()
    homography = np.zeros((3,3))

    sample_size = 4
    sample_count = 0
    prob_free_outliers = 0.99
    outlier_ratio = 0.5

    iterations = MAX_ITERATIONS

    previous_msg_len = 0
    while iterations > sample_count:
        # Sample 4 random points
        sampled_points = [point_map[i] for i in np.random.choice(number_of_points, sample_size)]
        sample_count += 1
        # Compute homography 
        H = compute_normalized_dlt(sampled_points)
        # Compute inliers
        inliers = set()
        for i in range(number_of_points):
            if distance(point_map[i][0], H) < THRESHOLD:
                inliers.add((point_map[i][0][0], point_map[i][0][1], point_map[i][0][2], point_map[i][0][3]))
        # Update best inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            homography = H
            # Update iterations
            outlier_ratio = 1 - (len(inliers) / number_of_points)
            iterations = math.ceil(math.log(1 - prob_free_outliers) / math.log(1 - (1 - outlier_ratio)**sample_size))
        
        # Sampled more times than needed after finding a good set of inliers.
        if iterations < sample_count:
            iterations = sample_count
        
        # If the inliers proportion attend the percentage in MIN_INLIERS, stop.
        elif len(best_inliers) > number_of_points * MIN_INLIERS:
            print(f'Minimum Inliers ({MIN_INLIERS:.0%}) Reached!')
            break
        
        # If the maximum number of iterations is reached, stop.
        elif sample_count == MAX_ITERATIONS:
            print(f'Max Iterations ({MAX_ITERATIONS}) Reached!')
            break
        
        msg = 'Iteration {} of {} - [{:.1f}%]'.format(sample_count, iterations, sample_count * 100/iterations)
        print('\r' + ' ' * previous_msg_len, end='\r')
        print(msg, end='\r',flush=True)
        previous_msg_len = len(msg)
        time.sleep(0.001)

        
    
    homography = compute_normalized_dlt(np.array(list(best_inliers)).reshape(-1,1,4))

    end = time.time()
    print('\n----------------------REPORT----------------------')
    print('RANSAC Finished! - Homography Estimation')
    print(f'Elapsed Time: {end-start:.2f} seconds')
    print(f'Threshold Used: {THRESHOLD}')
    print(f'Best Inlier Set: {len(best_inliers)} points')
    print(f'Best Inlier Ratio: {len(best_inliers)/number_of_points:.2%}')
    print(f'Outlier Ratio: {outlier_ratio:.2%}')
    print(f'Number of Iterations: {sample_count}')
    print(f'Final Homography Matrix:')
    print(homography)
    print('-------------------------------------------------')

    return homography, best_inliers


np.set_printoptions(precision=2, suppress=True)
# Exemplo de Teste da função de homografia usando o SIFT
MIN_MATCH_COUNT = 10

SET_NUMBER = 8
IMG_SELECT = (1,4)
RANSAC_THRESHOLD = 2.5

STRING_SET = f'./images/set{SET_NUMBER}/'
img1 = cv.imread(STRING_SET + f'{IMG_SELECT[0]}.jpg', 0)   # queryImage
img2 = cv.imread(STRING_SET + f'{IMG_SELECT[1]}.jpg', 0)   # trainImage


# Inicialização do SIFT
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

    point_map = np.array([
        [kp1[m.queryIdx].pt[0], 
         kp1[m.queryIdx].pt[1], 
         kp2[m.trainIdx].pt[0],
         kp2[m.trainIdx].pt[1]] for m in good
    ]).reshape(-1, 1, 4)

    try:
        M, best_inliers = ransac(point_map, THRESHOLD=RANSAC_THRESHOLD)

    except KeyboardInterrupt: 
        print(f'User Canceled the Program... Exiting')
        exit(0)

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   flags = 2)

img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

inlier_matches = []
for m in good:
    pt1 = (kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1])
    pt2 = (kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1])
    if (pt1[0], pt1[1], pt2[0], pt2[1]) in best_inliers:
        inlier_matches.append(m)

img_inliers = cv.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, **draw_params)

fig1, _ = plt.subplots(1, 2, figsize=(10, 7))
fig1.add_subplot(1, 2, 1)
plt.imshow(img3, 'gray')
plt.title('Matching (Without RANSAC)')
fig1.add_subplot(1, 2, 2)
plt.imshow(img_inliers, 'gray')
plt.title('Matching (With RANSAC)')

for i in range(len(fig1.axes)):
    fig1.axes[i].set_axis_off()

fig2, _ = plt.subplots(2, 2, figsize=(10, 7))
fig2.add_subplot(2, 2, 1)
plt.title('1st img')
plt.imshow(img1, 'gray')
fig2.add_subplot(2, 2, 2)
plt.title('2nd img')
plt.imshow(img2, 'gray')
fig2.add_subplot(2, 2, 3)
plt.title('1st after transformation')
plt.imshow(img4, 'gray')

for i in range(len(fig2.axes)):
    fig2.axes[i].set_axis_off()


try:
    plt.show(block=False)
    input('\nPress ENTER to End Program...')

except KeyboardInterrupt: pass

finally:
    print('Exiting...')
    cv.destroyAllWindows()
    plt.close('all')
    exit(0)