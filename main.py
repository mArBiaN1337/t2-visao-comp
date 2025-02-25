# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: Marcellus T. Biancardi

from typing import Tuple
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
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

'''
def compute_dlt(pts1 : np.ndarray, pts2 : np.ndarray) -> np.ndarray:
    pairs = np.array([(pts1[0,i], pts1[1,i], pts2[0,i], pts2[1,i]) for i in range(pts1.shape[1])])

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
'''

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

# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)
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

# Função do RANSAC
# Entradas:
# pts1: pontos da primeira imagem
# pts2: pontos da segunda imagem 
# dis_threshold: limiar de distância a ser usado no RANSAC
# N: número máximo de iterações (pode ser definido dentro da função e deve ser atualizado 
#    dinamicamente de acordo com o número de inliers/outliers)
# Ninl: limiar de inliers desejado (pode ser ignorado ou não - fica como decisão de vocês)
# Saídas:
# H: homografia estimada
# pts1_in, pts2_in: conjunto de inliers dos pontos da primeira e segunda imagens
def ransac(point_map : np.ndarray, THRESHOLD : float = 0.7, MAX_ITERATIONS : int = 1000, MIN_INLIERS : float = 0.8) -> np.ndarray:
    
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
    
    while iterations > sample_count:
        # Sample 4 random points
        sampled_points = [point_map[i] for i in np.random.choice(number_of_points, sample_size)]
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
            if len(best_inliers) > number_of_points * MIN_INLIERS:
                break
        
        sample_count += 1
        print('\r', end='', flush=True)
        print(f'\rIteration {sample_count} of {iterations} - ({sample_count/iterations:.2%})', end='', flush=True)
        time.sleep(0.01)
        
    
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
    return homography



np.set_printoptions(precision=2, suppress=True)
# Exemplo de Teste da função de homografia usando o SIFT
MIN_MATCH_COUNT = 10

set_number = 1
img_select = (1,2)

string_set = f'./images/set{set_number}/'
img1 = cv.imread(string_set + f'{img_select[0]}.jpg', 0)   # queryImage
img2 = cv.imread(string_set + f'{img_select[1]}.jpg', 0)   # trainImage


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

    M = ransac(point_map)

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   flags = 2)

img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(10, 7))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
plt.title('Matching (Without RANSAC)')
fig.add_subplot(2, 2, 2)
plt.title('1st img')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('2nd img')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('1st after transformation')
plt.imshow(img4, 'gray')
plt.show()

