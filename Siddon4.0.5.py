from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
import math
from numba import jit
import timeit
from scipy import sparse
import seaborn as sns
from sklearn.linear_model import Lasso

# image = imread(data_dir + "/phantom.png", as_grey=True)
# image = imread("bubbles.png", as_grey=True)  # 20180810
# image = imread("L.png", as_grey=True)  # 20180920
image = imread("dot3.png", as_grey=True)  # 20181109
# image = rescale(image, scale=0.4, mode='reflect')
image = rescale(image, scale=0.5, mode='reflect')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
# theta = np.linspace(0., 360., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta, circle=True)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

fig.tight_layout()
plt.show()

# --Calc reconstruction
# -Parameters
# radian
theta_rad = theta * ( np.pi / 180.0 )

# Pixel pitch of original image (dx, dy)
dx = 1.0
dy = 1.0

# Pixel pitch of detector (du)
du = 1.0

# x-,y-plane Numbers of original image (Nx, Ny)
# Nx、Nyはそれぞれx面、y面の数なので、画素数+1となることに注意。
Nx = image.shape[0] + 1
Ny = image.shape[1] + 1

# Pixel Number of Detector (Nk)
Nk = max(image.shape)

# Data points of theta (Nc)
Nc = theta.shape[0]

# Origin of original image (bx, by)
# "//"は切り捨て除算
# 20180902 - begin # 原点も0.5だけずらす必要あり
bx = -(Nx-1) // 2 + 0.5
by = -(Ny-1) // 2 + 0.5
# bx = -(Nx-1) // 2
# by = -(Ny-1) // 2
# 20180902 - end

# Number of cross points for each line (Nv[k][theta]) - reshape after
Nv = np.empty

# System matrix (L) - reshape after
# 20180708 - begin
# 「画素数」は(Nx-1)*(Ny-1)
sL = np.zeros( (Nx-1) + (Ny-1) )
L = np.zeros( (Nk * Nc, (Nx-1) * (Ny-1) ) )
Lsp = sparse.lil_matrix( (Nk * Nc, (Nx-1) * (Ny-1) ) )
# 20180708 - end

# Reconstructed image (P[Nv*theta.shape[0]]) - reshape after
# 20180708 - begin
# 「画素数」は(Nx-1)*(Ny-1)
rho = np.zeros( (Nk, (Nx-1) * (Ny-1) ) )
rho_sp = np.zeros( (Nk,  (Nx-1) * (Ny-1) ) )
reconIm = np.zeros( ( (Nx-1), (Ny-1) ) )
reconIm_SIRT = np.zeros( ( (Nx-1), (Ny-1) ) )   # 20180917
reconIm_part = np.zeros( ( (Nx-1), (Ny-1) ) )
# 20180708 - end

# 呼び出されたインデックスの回数を保存
counterMap = np.zeros( (Nx, Ny) )

# Crosspoint specify array of x,y (alpha_x[k][theta][Nx], alpha_y[k][theta][Ny])
alpha_x = np.zeros(Nx)
alpha_y = np.zeros(Ny)
# 20180728 - begin
alpha_align_x = np.empty
alpha_align_y = np.empty
# 20180728 - end

# Crosspoint specify array (alpha_xy[k][theta][Nv]) - reshape after
alpha_xy = np.empty

# index
i_min = np.empty
i_max = np.empty
j_min = np.empty
j_max = np.empty

# index of reconstructed image data
# 20180708 - begin
i_m = np.zeros(Nx-1 + Ny-1)
j_m = np.zeros(Nx-1 + Ny-1)
# 20180708 - end

# Length btw. p1 and p2
d_conv = np.empty

# 光線・検出器の位置ベクトル確認用
chk1x = np.zeros((Nk, Nc))
chk1y = np.zeros((Nk, Nc))
chk2x = np.zeros((Nk, Nc))
chk2y = np.zeros((Nk, Nc))

# 光線とボクセルの交点を可視化する
crosspointsX = np.zeros( (Nk * Nc, Nx + Ny) )
crosspointsY = np.zeros( (Nk * Nc, Nx + Ny) )

# 20180707 - begin
print("sinogram.shape", sinogram.shape)
# 20180707 - end

# 関数定義
def pPos(alpha, p1, p2):
    return p1 + alpha * ( p2 - p1 )

def phi(alpha, p1, p2, b, d):
    return ( pPos(alpha, p1, p2) - b) / d

@jit
def calc_system_matrix(theta_rad, dx, dy, du, Nx, Ny, Nk, Nc, bx, by, Nv, sL, \
                       alpha_x, alpha_y, alpha_xy, i_min, i_max, j_min, j_max, i_m, j_m, \
                       d_conv, chk1x, chk1y, chk2x, chk2y, \
                       crosspointsX, crosspointsY):
    L = np.zeros( (Nk * Nc, (Nx-1) * (Ny-1) ) )  # 20180708
    #     L = np.zeros( (Nk * Nc, Nx * Ny) )
    for k in range(Nk):
        for t in range(Nc):
            # 回転行列
            U = np.matrix([[np.cos(theta_rad[t]), -np.sin(theta_rad[t])], [np.sin(theta_rad[t]), np.cos(theta_rad[t])]])
#             U = np.matrix([[np.cos(-theta_rad[t]), -np.sin(-theta_rad[t])], [np.sin(-theta_rad[t]), np.cos(-theta_rad[t])]])   # 20180920
# 20180902 - begin
#             p1x = float(np.dot(U, np.matrix( [[ (k-Nk/2.0)*du + du/2.0 ], [ -(Ny-1)*dy + dy/2.0 ]] ))[0])
#             p1y = float(np.dot(U, np.matrix( [[ (k-Nk/2.0)*du + du/2.0 ], [ -(Ny-1)*dy + dy/2.0 ]] ))[1])
# 20180902 - end
# 20180920 - begin  # p1とp2の位置関係が逆になっていたので修正。サイノグラムとシステム行列の検出器のインデックスの方向が逆だったので修正。
#             p1x = float(np.dot(U, np.matrix( [[ (Nk/2.0 - (k+1) )*du + du/2.0 ], [ -(Ny-1)*dy + dy/2.0 ]] ))[0])
#             p1y = float(np.dot(U, np.matrix( [[ (Nk/2.0 - (k+1) )*du + du/2.0 ], [ -(Ny-1)*dy + dy/2.0 ]] ))[1])
# 20180920 - end
# 20181109 - begin  # p1とp2の位置関係が逆になっていたので修正。
            p1x = float(np.dot(U, np.matrix( [[ (k-Nk/2.0)*du + du/2.0 ], [ (Ny-1)*dy + dy/2.0 ]] ))[0])
            p1y = float(np.dot(U, np.matrix( [[ (k-Nk/2.0)*du + du/2.0 ], [ (Ny-1)*dy + dy/2.0 ]] ))[1])
# 20181109 - end
            chk1x[k][t] = p1x
            chk1y[k][t] = p1y
# 20180902 - begin # y方向にも0.5だけずらさなければならないのでは? 
            p2x = float(np.dot(U, np.matrix([[(k-Nk/2.0)*du + du/2.0 ], [ (Ny-1)*dy + dy/2.0 ]]))[0])
            p2y = float(np.dot(U, np.matrix([[(k-Nk/2.0)*du + du/2.0 ], [ (Ny-1)*dy + dy/2.0 ]]))[1])
# 20180902 - end
# 20180920 - begin
#             p2x = float(np.dot(U, np.matrix([[(Nk/2.0 - (k+1) )*du + du/2.0 ], [ (Ny-1)*dy + dy/2.0 ]]))[0])
#             p2y = float(np.dot(U, np.matrix([[(Nk/2.0 - (k+1) )*du + du/2.0 ], [ (Ny-1)*dy + dy/2.0 ]]))[1])
# 20180920 - end
# 20181109 - begin  # p1とp2の位置関係が逆になっていたので修正。
            p2x = float(np.dot(U, np.matrix([[(k-Nk/2.0)*du + du/2.0 ], [ -(Ny-1)*dy + dy/2.0 ]]))[0])
            p2y = float(np.dot(U, np.matrix([[(k-Nk/2.0)*du + du/2.0 ], [ -(Ny-1)*dy + dy/2.0 ]]))[1])
# 20181109 - end
            chk2x[k][t] = p2x
            chk2y[k][t] = p2y

            # p1x==p2x,p1y==p2yの場合はαの分母が0になってしまうので処理を分ける。
            if p1x != p2x and p1y != p2y:
                alpha_x = ( bx + np.array(np.arange(Nx)) * dx - p1x ) / ( p2x - p1x )
                alpha_y = ( by + np.array(np.arange(Ny)) * dy - p1y ) / ( p2y - p1y ) 
                alpha_xmin = np.min((alpha_x[0], alpha_x[Nx-1]))
                alpha_xmax = np.max((alpha_x[0], alpha_x[Nx-1]))
                alpha_ymin = np.min((alpha_y[0], alpha_y[Ny-1]))
                alpha_ymax = np.max((alpha_y[0], alpha_y[Ny-1]))
                alpha_min = np.max((alpha_xmin, alpha_ymin))
                alpha_max = np.min((alpha_xmax, alpha_ymax))
                if p1x < p2x:
                    if alpha_min == alpha_xmin:
                        i_min = 1
                    else:
                        i_min = math.ceil( phi(alpha_min, p1x, p2x, bx, dx) )  # 20180723
                    if alpha_max == alpha_xmax:
                        i_max = Nx - 1
                    else:
                        i_max = math.floor( phi(alpha_max, p1x, p2x, bx, dx) )
                    alpha_align_x = alpha_x[ i_min : i_max+1 ]  # 20180728
                    
                else: # p1x > p2x
                    if alpha_min == alpha_xmin:
                        i_max = Nx - 2
                    else:
                        i_max = math.floor( phi( alpha_min, p1x, p2x, bx, dx) )
                    if alpha_max == alpha_xmax:
                        i_min = 0
                    else:
                        i_min = math.ceil( phi( alpha_max, p1x, p2x, bx, dx) )
                    # 20180728 - begin
                    temp_alpha_x = alpha_x[ i_min : i_max+1 ]
                    alpha_align_x = temp_alpha_x[::-1]
                    # 20180728 - end                    

                if p1y < p2y:
                    if alpha_min == alpha_ymin:
                        j_min = 1
                    else:
                        j_min = math.ceil( phi(alpha_min, p1y, p2y, by, dy) )  # 20180723
                    if alpha_max == alpha_ymax:
                        j_max = Ny - 1
                    else:
                        j_max = math.floor( phi(alpha_max, p1y, p2y, by, dy) )
                    alpha_align_y = alpha_y[ j_min : j_max+1 ] # 20180728

                else: # p1y > p2y
                    if alpha_min == alpha_ymin:
                        j_max = Ny - 2
                    else:
                        j_max = math.floor( phi( alpha_min, p1y, p2y, by, dy) )
                    if alpha_max == alpha_ymax:
                        j_min = 0
                    else:
                        j_min = math.ceil( phi( alpha_max, p1y, p2y, by, dy) )
                    # 20180728 - begin
                    temp_alpha_y = alpha_y[ j_min : j_max+1 ]
                    alpha_align_y = temp_alpha_y[::-1]
                    # 20180728 - end

                # αx、αyのi_min～i_max、j_min～j_maxのみ切り出したものを結合し、昇順にソート(sort)する。
                # 20180728 - begin
                alpha_xy = np.sort( np.unique( np.append( np.concatenate( (alpha_align_x, alpha_align_y), axis=0 ), alpha_min ) ) ) # 20180811
                # 20180728 - end
                Nv = len(alpha_xy)
                # 20180805 - begin
                alpha_ave  = ( alpha_xy[1 : Nv] + alpha_xy[0 : Nv - 1] ) / 2.0
                alpha_diff = ( alpha_xy[1 : Nv] - alpha_xy[0 : Nv - 1] )
                # 【メモ】i_min,i_maxを使っていない。i_m[0] = i_min, i_m[Nv-1] = i_maxか？
                #【メモ】αxyを作成するときにsortにかけているので、αx、αyをp1、p2の大小で並び順を直した意味がないのでは？
                i_m = np.floor( phi(alpha_ave, p1x, p2x, bx, dx) )
                j_m = np.floor( phi(alpha_ave, p1y, p2y, by, dy) )
                i_m = i_m.astype( np.int64 )
                j_m = j_m.astype( np.int64 )
                d_conv = np.sqrt( (p2x - p1x)**2 + (p2y - p1y)**2 )
                sL = alpha_diff * d_conv
                # 20180805 - end

            elif p1x == p2x:
                alpha_y = ( by + np.array(np.arange(Ny)) * dy - p1y ) / ( p2y - p1y )        
                # 20180723 - begin
                alpha_ymin = np.min( (alpha_y[0], alpha_y[Ny-1]) )
                alpha_ymax = np.max( (alpha_y[0], alpha_y[Ny-1]) )
                # 20180723 - end
                alpha_min = alpha_ymin
                alpha_max = alpha_ymax
                # 20180726 - begin
                if p1y < p2y:
                    j_min = 1
                    j_max = Ny - 1
                    alpha_align_y = alpha_y[ j_min : j_max+1 ]  # 20180728
                else:
                    j_min = 0
                    j_max = Ny - 2
                    # 20180728 - begin
                    temp_alpha_y = alpha_y[ j_min : j_max+1 ]
                    alpha_align_y = temp_alpha_y[::-1]
                    # 20180728 - end
                # 20180726 - end
                # 20180728 - begin
                alpha_xy = np.sort( np.append( alpha_align_y, alpha_min ) )
                # 20180728 - end
                Nv = len(alpha_xy)
                # 20180805 - begin
                alpha_ave  = ( alpha_xy[1 : Nv] + alpha_xy[0 : Nv - 1] ) / 2.0
                alpha_diff = ( alpha_xy[1 : Nv] - alpha_xy[0 : Nv - 1] )
                # 【メモ】i_min,i_maxを使っていない。i_m[0] = i_min, i_m[Nv-1] = i_maxか？
                #【メモ】αxyを作成するときにsortにかけているので、αx、αyをp1、p2の大小で並び順を直した意味がないのでは？
                i_m = np.floor( ( p1x * np.ones( len(alpha_ave) ) - bx ) / dx )
                j_m = np.floor( phi(alpha_ave, p1y, p2y, by, dy) )
                i_m = i_m.astype( np.int64 )
                j_m = j_m.astype( np.int64 )
                d_conv = np.sqrt( (p2x - p1x)**2 + (p2y - p1y)**2 )
                sL = alpha_diff * d_conv
                # 20180805 - end
        
            elif p1y == p2y:
                alpha_x = ( bx + np.array(np.arange(Nx)) * dx - p1x) / ( p2x - p1x )
                # 20180723 - begin
                alpha_xmin = np.min( (alpha_x[0], alpha_x[Nx-1]) )
                alpha_xmax = np.max( (alpha_x[0], alpha_x[Nx-1]) )
                # 20180723 - end
                alpha_min = alpha_xmin
                alpha_max = alpha_xmax
                # 20180726 - begin
                if p1x < p2x:
                    i_min = 1
                    i_max = Nx - 1
                    alpha_align_x = alpha_x[ i_min : i_max+1 ]  # 20180728
                else:
                    i_min = 0
                    i_max = Nx - 2
                    # 20180728 - begin
                    temp_alpha_x = alpha_x[ i_min : i_max+1 ]
                    alpha_align_x = temp_alpha_x[::-1]
                    # 20180728 - end
                # 20180726 - end
                # 20180728 - begin
                alpha_xy = np.sort( np.append( alpha_align_x, alpha_min ) )
                # 20180728 - end
                Nv = len(alpha_xy)
                # 20180805 - begin
                alpha_ave  = ( alpha_xy[1 : Nv] + alpha_xy[0 : Nv - 1] ) / 2.0
                alpha_diff = ( alpha_xy[1 : Nv] - alpha_xy[0 : Nv - 1] )
                # 【メモ】i_min,i_maxを使っていない。i_m[0] = i_min, i_m[Nv-1] = i_maxか？
                #【メモ】αxyを作成するときにsortにかけているので、αx、αyをp1、p2の大小で並び順を直した意味がないのでは？
                i_m = np.floor( phi(alpha_ave, p1x, p2x, bx, dx) )
                j_m = np.floor( ( p1y * np.ones( len(alpha_ave) ) - by ) / dy )
                i_m = i_m.astype( np.int64 )
                j_m = j_m.astype( np.int64 )
                d_conv = np.sqrt( (p2x - p1x)**2 + (p2y - p1y)**2 )
                sL = alpha_diff * d_conv
                # 20180805 - end
            
            # 光線とボクセルの交点を可視化するために交点座標を保存 - [0]
            # 要素数160以下のalpha_xyを要素数161*161のcrosspointsX/Yに代入するためにパディング
            crosspointsX_ = p1x + alpha_xy * ( p2x - p1x )
            crosspointsY_ = p1y + alpha_xy * ( p2y - p1y )
            crosspointsX[ Nc * k + t ][:] = np.pad( crosspointsX_, (0, Nx + Ny - len(crosspointsX_)), 'constant', constant_values=0 )
            crosspointsY[ Nc * k + t ][:] = np.pad( crosspointsY_, (0, Nx + Ny - len(crosspointsY_)), 'constant', constant_values=0 )
            # - [0]
            ssL = np.zeros( (Nx-1) * (Ny-1) )  # 20180708
            for m in range( len(sL) ):
                ssL[ (Nx-1) * i_m[m] + j_m[m] ] = sL[m]  # 20180708        
            L[ Nc * k + t ][:] = ssL

    return L
            
# システム行列の計算
print( calc_system_matrix(theta_rad, dx, dy, du, Nx, Ny, Nk, Nc, bx, by, Nv, sL, \
                       alpha_x, alpha_y, alpha_xy, i_min, i_max, j_min, j_max, i_m, j_m, \
                       d_conv, chk1x, chk1y, chk2x, chk2y, \
                       crosspointsX, crosspointsY).shape )
L = calc_system_matrix(theta_rad, dx, dy, du, Nx, Ny, Nk, Nc, bx, by, Nv, sL, \
                       alpha_x, alpha_y, alpha_xy, i_min, i_max, j_min, j_max, i_m, j_m, \
                       d_conv, chk1x, chk1y, chk2x, chk2y, \
                       crosspointsX, crosspointsY)

timecheck = timeit.timeit( "calc_system_matrix(theta_rad, dx, dy, du, Nx, Ny, Nk, Nc, bx, by, Nv, sL, \
                      alpha_x, alpha_y, alpha_xy, i_min, i_max, j_min, j_max, i_m, j_m, \
                      d_conv, chk1x, chk1y, chk2x, chk2y, \
                      crosspointsX, crosspointsY)", globals=globals(), number=1 )
print( "time of calc_system_matrix",  timecheck )

# 疎行列に変換
Lsp = sparse.lil_matrix(L)
Lsp = Lsp.tocsc()
LspT = Lsp.T.tocsr()

# system行列を可視化してみる
x = np.arange(Nx * Ny)
y = np.arange(Nc)
X, Y = np.meshgrid(x, y)
plt.figure(figsize=(10,10))
sns.heatmap(Lsp.toarray()[Nc * 80 : Nc * 81, :(Nx*Ny)])

# 光線とボクセルの交点を可視化 - start
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1,1,1)
for t in range(Nc):
    ax.scatter(crosspointsX[ Nc * 80 + t ][:], crosspointsY[ Nc * 80 + t ][:], marker=".")
ax.scatter(chk1x[80][:], chk1y[80][:], marker=".")
ax.scatter(chk2x[80][:], chk2y[80][:], c='orange', marker=".")
ax.set_xlabel('x')
ax.set_ylabel('y')            
# 光線とボクセルの交点を可視化 - end

# 表示して確認 - start
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(chk1x[0][:], chk1y[0][:], marker=".")
ax.scatter(chk2x[0][:], chk2y[0][:], c='orange', marker=".")
ax.set_title('k=0 scatter plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('1.0')
# 表示して確認 - end 

# 光源と検出器の位置がずれていないか確認 - start
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1,1,1)
ax.scatter(crosspointsX[Nc * 60 + 0][:], crosspointsY[ Nc * 60 + 0 ][:], marker=".")
ax.scatter(chk1x[80][:], chk1y[80][:], marker=".")
ax.scatter(chk2x[80][:], chk2y[80][:], c='orange', marker=".")
ax.set_xlabel('x')
ax.set_ylabel('y')            
# 光線とボクセルの交点を可視化 - end

# ----

#20180917 - SIRTの実装
def SIRT(W, p, Nm, Nn, Ni):
    # Ni : Number of iteration
    R = sparse.lil_matrix((Nm, Nm))
    C = sparse.lil_matrix((Nn, Nn))
    
    W_rowsum = W.sum(axis=0)
    W_colsum = W.sum(axis=1)
    R_temp =  np.ravel( 1 / W_colsum )
    C_temp =  np.ravel( 1 / W_rowsum )
    R = sparse.diags(R_temp, 0).tocsr()
    C = sparse.diags(C_temp, 0).tocsr()
    
    v = np.zeros(Nn)
    v = sparse.lil_matrix(v).tocsr()
    for i  in range(Ni-1):
        WPD = R.dot( p.T - W.dot(v.T) ).tocsr()  # Weighted Projection Difference
        WBP = C.dot( W.T.dot(WPD) ).tocsr()  # Weighted BackProjection
        v.T = v.T + WBP
        
    return v.T

Ni = 100

y = sparse.lil_matrix( sinogram.flatten() ).tocsc()
x = sparse.lil_matrix( np.zeros( Lsp.shape[1] ) ).tocsr()
x = SIRT(Lsp, y, Lsp.shape[0], Lsp.shape[1], Ni)

reconIm_SIRT = np.reshape(x.todense(), (Nx-1, Ny-1))

# 回転角方向に間引く(LASSOとの比較のため) - begin
Nstep_c_SIRT = 10
abstL_SIRT = Lsp[:Nc][:]
partL_c_SIRT = abstL_SIRT[::Nstep_c_SIRT][:]
part_y_c_SIRT = sparse.lil_matrix( np.ravel( sinogram[0][::Nstep_c_SIRT]) ).tocsc()

for k in range( 1, Nk ):
    # 一旦、Nc行分だけ抜き出す
    abstL_SIRT = Lsp[ Nc * k : Nc * ( k + 1 ) ][:]
    # 間引く
    tempL_c_SIRT = abstL_SIRT[::Nstep_c_SIRT][:]
    partL_c_SIRT = sparse.vstack( ( partL_c_SIRT, tempL_c_SIRT ) )
    
    abst_y_SIRT = sparse.lil_matrix( sinogram[k][::Nstep_c_SIRT] ).tocsc()
    part_y_c_SIRT = sparse.hstack( ( part_y_c_SIRT, abst_y_SIRT ) )

x0_c_SIRT = partL_c_SIRT.T.dot(part_y_c_SIRT.T).tocsc()
x_c_SIRT = x0_c_SIRT
x_part_c_SIRT = SIRT(partL_c_SIRT, part_y_c_SIRT, partL_c_SIRT.shape[0], partL_c_SIRT.shape[1], Ni)
reconIm_part_c_SIRT = np.reshape( x_part_c_SIRT.todense(), (Nx-1, Ny-1) )
# 回転角方向に間引く(LASSOとの比較のため) - end

# ２次元に戻す
plt.figure( figsize=(8, 8) )
plt.pcolor(np.array(reconIm_SIRT.T))

plt.figure( figsize=(8, 8) )
plt.pcolor(np.array(reconIm_part_c_SIRT.T))

# ---

# 最小二乗法のiteration回数の指定  # 20180717
Ni = 1000

# 検出器のラベルごとに分解して線型方程式の解を求める
for k in range(Nk):
    partL = Lsp[ Nc * k : Nc * k + Nc ][:]   #csc
    d = sinogram[k][:].flatten()

    # 20180717 - begin
    X = sparse.linalg.lsmr(partL, d, damp=0.0, atol=0.0, btol=0.0, conlim=100000000.0, maxiter=None, show=False, x0=None)
    # 20180718 - end

    rho[k][:] = X[0]    

    # 2次元に戻す
    reconIm = reconIm + np.reshape( rho[k][:], (Ny-1, Nx-1) )  # 20180709
    
# 結果を表示
plt.figure( figsize=(8, 8) )
plt.pcolor(reconIm.T)

# ---

# 軟判定しきい値関数
def SoftThr(x,lam):
    z = np.zeros(x.shape).astype(np.float32)
    mask = x > lam
    z = z + (x - lam) * mask
    mask = x < -lam
    z = z + (x + lam) * mask
    return z

# LASSOの誤差関数
def loss_LASSO(A, x, y, lam):
    v = y.T - A.dot(x).tocsr() # CSR
    cost = sparse.linalg.norm(v, ord='fro')**2 # Frobenius norm (Same as ord=2 in numpy.linalg.norm)
    cost = 0.5 * cost + lam * sparse.linalg.norm(x, ord=1)
    return cost

def ISTA(A, x, y, lam, L0):  # Assume A,y,x are CSC
    kx = x  # CSC
    v = y.T - A.dot(x).tocsr()  # CSR
    ktemp = 0.5 * sparse.linalg.norm(v, ord='fro')**2
    L = L0

    grad_x = -A.T.dot(v).tocsc() / lam  # CSC
    MMx = x - grad_x / L # CSC
    MMv = y.T - A.dot(MMx).tocsr() # CSR
    temp = 0.5 * sparse.linalg.norm(MMv, ord='fro')**2
    Diffx = MMx - kx # CSC

    MMtemp = ktemp + grad_x.T.dot(Diffx.tocsr()).todense() + 0.5 * L * sparse.linalg.norm(Diffx, ord='fro')**2
    # ラインサーチ法を入れるとMMx=xとなるまでMMxを更新してしまう。
#     while MMtemp < temp:
#         L = L * 1.1
#         MMx = x - grad_x / L  # CSC
#         print("<1.1> x, MMx", x[0][0], MMx[0][0])
#         temp = 0.5 * sparse.linalg.norm(y.T - A.dot(MMx), ord='fro')**2
#         Diffx2 = MMx - kx  # CSC
#         MMtemp = ktemp + grad_x.T.dot(Diffx2.tocsr()).todense() + 0.5 * L * sparse.linalg.norm(Diffx2, ord='fro')**2

    # SoftThrの計算が間違っていそう
    # Matrix型の変数を与えているので、ndarrayで計算できていたアルゴリズムが正しく動いていない？
    # -> 正解。SoftThrに与える時にxをtodense()でmatrix型として与えてしまっていた。toarray()で与えることでndarray型で渡すことができる。
    # 20181123 - begin
    x_out = SoftThr(MMx.toarray(), 1 / L)
    x_out = sparse.lil_matrix(x_out).tocsc()
#     x = SoftThr(MMx.toarray(), 1 / L)
#     x = sparse.lil_matrix(x).tocsc()
    # 20181123 - end
    
    return x_out


# ISTAで値を更新してLASSOの誤差関数で評価。ある程度まで誤差が小さくなったら、解が得られたものとする。
EPSILON = 10**(-5)
y = sparse.lil_matrix( sinogram.flatten() )
y = y.tocsc()  # CSC

err = 1.0
# Initial condition
x0 = LspT.dot(y.T).tocsc() # LspT:CSR, y.T:CSR, x:CSC
x = x0
lam = 0.1  # lambda
P0 = 10**6  # Lipschitz定数の初期値

# ISTAを繰り返す回数を指定
Nt = 5000

# 結果とコストを格納する配列
cost = np.zeros(Nt)

# 20180930 temporarily comment out
for t in range(Nt):
    out = ISTA(Lsp, x, y, lam, P0)
    x = out.tocsr()
    cost[t] = 0.5 * sparse.linalg.norm(y.T - Lsp.dot(x).tocsr() , ord='fro')**2

reconIm = np.reshape(x.todense(), (Nx-1, Ny-1))   # 20180723
# 20180930 temporarily comment out

# 一部の検出器のデータのみで再構成
# 20181120 temporarily comment out
# kの値を間引く
Nstep_k = 2
partL = Lsp[ : Nc ][:]
part_y = sparse.lil_matrix( np.ravel( sinogram[0][:] ) ).tocsc()

for k in range( 1, Nk // Nstep_k ):
    tempL = Lsp[ Nc * ( Nstep_k * k ) : Nc * ( Nstep_k * k + 1 ) ][:]
    partL = sparse.vstack( ( partL, tempL ) )
    temp_y = sparse.lil_matrix( np.ravel( sinogram[ Nstep_k * k ][:] ) ).tocsc()
    part_y = sparse.hstack( (part_y, temp_y) )
x0 = partL.T.dot(part_y.T).tocsc() # LspT:CSR, y.T:CSR, x:CSC
x = x0
# # 20181020 - begin
for t in range(Nt):
    out = ISTA(partL, x, part_y, lam, P0)
    out = out.tocsr()
    x = out
# 20181020 - end
reconIm_part = np.reshape( out.todense(), (Nx-1, Ny-1) )
# 20181120 temporarily comment out

# 20181123 temporarily comment out
# 20181114 - begin
# 回転角方向を間引く
print("Calc Angular Decimation")

Nstep_c = 10
cost_c = np.zeros(Nt)

abstL = Lsp[:Nc][:]
partL_c = abstL[::Nstep_c][:]
part_y_c = sparse.lil_matrix( np.ravel( sinogram[0][::Nstep_c]) ).tocsc()

for k in range( 1, Nk ):
    # 一旦、Nc行分だけ抜き出す
    abstL = Lsp[ Nc * k : Nc * ( k + 1 ) ][:]
    # 間引く
    tempL_c = abstL[::Nstep_c][:]
    partL_c = sparse.vstack( ( partL_c, tempL_c ) )
    
    abst_y = sparse.lil_matrix( np.ravel( sinogram[k][::Nstep_c] ) ).tocsc()
    part_y_c = sparse.hstack( ( part_y_c, abst_y ) )

x0_c = partL_c.T.dot(part_y_c.T).tocsc()
x_c = x0_c
for t in range(Nt):
    out_c = ISTA(partL_c, x_c, part_y_c, lam, P0)
    out_c = out_c.tocsr()
    x_c = out_c
    cost_c[t] = 0.5 * sparse.linalg.norm(part_y_c.T - partL_c.dot(x_c).tocsr() , ord='fro')**2
    
reconIm_part_c = np.reshape( out_c.todense(), (Nx-1, Ny-1) )
# 20181123 temporarily comment out

# for k in range(81):
#     partL = Lsp[ Nc * k : Nc * k + Nc ][:]   #csc
#     d = sparse.lil_matrix( sinogram[k][:].flatten() )
#     d = d.tocsc()
#     for t in range(Nt):
#         x = ISTA(partL, x, d, lam, P0)
#         x = x.tocsr()
#     rho_sp[k][:] = x.toarray().flatten()
    
# reconIm_part = reconIm_part + np.reshape( rho_sp[80][:], (Ny-1, Nx-1) )

# 結果を表示
# コスト関数の変化を描画
plt.figure()
plt.plot(np.arange(Nt), cost, marker="o")

# ２次元に戻す
plt.figure( figsize=(8, 8) )
plt.pcolor(np.array(reconIm.T))

# 検出器を間引き
plt.figure( figsize=(8, 8) )
plt.pcolor(np.array(reconIm_part.T))
# plt.pcolor(np.array(np.reshape( rho_sp[80][:], (Ny-1, Nx-1) ).T))  # 20180723

# コスト関数の変化を描画
plt.figure()
plt.plot(np.arange(Nt), cost_c, marker="o")

# 回転角を間引き
plt.figure( figsize=(8, 8) )
plt.pcolor(np.array(reconIm_part_c.T))

# ---

# 20181222 - ISTA from Qiita
def solve_lasso(A, y, alpha, maxiter=100, tol=1.0e-4):
    """ Solve lasso problem """
    x0 = np.zeros(A.shape[1])
    rho = supermum_eigen(A.T @ A)
    x = []
    for it in range(maxiter):
        x_new = update(x0, A, y, alpha, rho)
#         if (np.abs(x0 - x_new) < tol).all():
#             return x_new
        x0 = x_new
#     raise ValueError('Not converged.')
    return x_new   # 一定回数計算したら結果を返すように変更。

def update(x0, A, y, alpha, rho):
    """ Make an iteration with given initial guess x0 """
    res = y - A @ x0
    return soft_threashold(x0 + (A.T @ res) / rho, alpha / rho)

def soft_threashold(y, alpha):
    return np.sign(y) * np.maximum(np.abs(y) - alpha, 0.0)

def supermum_eigen(A):
    return np.max(np.sum(np.abs(A), axis=0))

# full data
y = sinogram.flatten()

alpha = 0.1
out = solve_lasso(L, y, alpha, tol=1e-5, maxiter=1000000)
reconIm = np.reshape(out, (Nx-1, Ny-1))

# Angular sparse
Nstep_c = 10

abstL = L[:Nc][:]
partL_c = abstL[::Nstep_c][:]
part_y_c = np.ravel( sinogram[0][::Nstep_c])

for k in range( 1, Nk ):
    # 一旦、Nc行分だけ抜き出す
    abstL = L[ Nc * k : Nc * ( k + 1 ) ][:]
    # 間引く
    tempL_c = abstL[::Nstep_c][:]
    partL_c = np.vstack( ( partL_c, tempL_c ) )
    
    abst_y = np.ravel( sinogram[k][::Nstep_c] )
    part_y_c = np.hstack( (part_y_c, abst_y) )

out_c = solve_lasso(partL_c, part_y_c, alpha, tol=1e-5, maxiter=1000)
reconIm_c = np.reshape(out_c, (Nx-1, Ny-1))

# ２次元に戻す
plt.figure( figsize=(8, 8) )
plt.pcolor(np.array(reconIm.T))

plt.figure( figsize=(8, 8) )
plt.pcolor(np.array(reconIm_c.T))

# ---

# scikit-learnによるLassoを利用
y = sinogram.flatten()

rgr_lasso = Lasso(alpha=0.001)
rgr_lasso.fit(L, y)
rec_l1 = rgr_lasso.coef_.reshape(Nx-1, Ny-1)

# 20181219 - begin
# 回転角方向を間引く
Nstep_c = 10

abstL = L[:Nc][:]
partL_c = abstL[::Nstep_c][:]
part_y_c = np.ravel( sinogram[0][::Nstep_c])

for k in range( 1, Nk ):
    # 一旦、Nc行分だけ抜き出す
    abstL = L[ Nc * k : Nc * ( k + 1 ) ][:]
    # 間引く
    tempL_c = abstL[::Nstep_c][:]
    partL_c = np.vstack( ( partL_c, tempL_c ) )
    
    abst_y = np.ravel( sinogram[k][::Nstep_c] )
    part_y_c = np.hstack( (part_y_c, abst_y) )
    
rgr_lasso_c = Lasso(alpha=0.001)
print(partL_c.shape, part_y_c.shape)
rgr_lasso_c.fit(partL_c, part_y_c)
rec_l1_c = rgr_lasso_c.coef_.reshape(Nx-1, Ny-1)
# 20181219 - end
    
# ２次元に戻す
plt.figure( figsize=(8, 8) )
plt.pcolor(np.array(rec_l1.T))

plt.figure( figsize=(8, 8) )
plt.pcolor(np.array(rec_l1_c.T))

# ---

# 軟判定しきい値関数
def SoftThr(x,lam):
    z = np.zeros(x.shape).astype(np.float32)
    mask = x > lam
    z = z + (x - lam) * mask
    mask = x < -lam
    z = z + (x + lam) * mask
    return z

# ADMM
# Initial condition
y = sparse.lil_matrix( sinogram.flatten() ).tocsc()
x = LspT.dot(y.T).tocsc()
z = x
u = sparse.lil_matrix( np.zeros( x.shape[0] ) ).T.tocsc()
lam = 0.1  # lambda
# 罰金項の係数の設定
mu = 1.0
# 予め固定値を計算
## C = sparse.linalg.inv( mu * sparse.identity( Lsp.shape[1] ) + 1/lam * Lsp.T.dot(Lsp) )
C = sparse.lil_matrix( np.linalg.inv( mu * np.identity( Lsp.shape[1] ) + 1/lam * LspT.dot(Lsp) ) )

# 繰り返し回数
Ni = 5

#ADMM更新式
for i in range(Ni):
    x = sparse.lil_matrix( C.dot( 1/lam * LspT.dot(y.T) + mu * (z - u) ) ).tocsc()
    z = SoftThr( x.toarray() + u.toarray(), 1/mu )
    z = sparse.lil_matrix(z).tocsc()
    u = u + x - z
    
reconIm = np.reshape( x.todense(), (Nx-1, Ny-1) )

# ２次元に戻す
plt.figure( figsize=(8, 8) )
plt.pcolor(np.array(reconIm.T))

# ---

#---疎行列を使わないバージョン---
'''
# LASSOの誤差関数
def loss_LASSO(A, x, y, lam):
    v = y - np.dot(A, x)
    cost = np.linalg.norm(v, ord=2)**2
    cost = 0.5 * cost + lam * np.linalg.norm(x, ord=1)
    return cost

# 軟判定しきい値関数
def SoftThr(x,lam):
    z = np.zeros(x.shape).astype(np.float32)
    mask = x > lam
    z = z + (x - lam) * mask
    mask = x < -lam
    z = z + (x + lam) * mask
    return z 

def ISTA(A, x, y, lam, L0):
    kx = x
    print("y, A, x", y.shape, A.shape, x.shape)
    v = y - np.dot(A, x)
    ktemp = 0.5 * np.linalg.norm(v, ord=2)**2
    L = L0

    # 勾配計算
    grad_x = -np.dot(A.T, v) / lam
    # 更新候補点へ移動
    MMx = x - grad_x / L
    MMv = y - np.dot(A, MMx)
    # コスト関数の一部を計算
    temp = 0.5 * np.linalg.norm(MMv, ord=2)**2
    # 現在点と更新候補点との差分を計算
    Diffx = MMx - kx

    # メジャライザーを計算
    MMtemp = ktemp + np.dot(grad_x, Diffx) + 0.5 * L * np.linalg.norm(Diffx, ord=2)**2
    print("temp", temp)
    print("ktemp", ktemp)
    print("grad_x * Diffx=", np.dot(grad_x, Diffx))
    print("0.5 * L * || Diffx ||**2", 0.5 * L * np.linalg.norm(Diffx, ord=2)**2)
    
    while MMtemp < temp:
        print("MMtemp, temp", MMtemp, temp)
        # リプシッツ定数を大きくする
        L = L * 1.1
        # 更新候補位置再計算
        MMx = x - grad_x / L
        # コスト関数の一部を再計算
        temp = 0.5 * np.linalg.norm(y - np.dot(A, MMx), ord=2)**2
        # 現在点と再計算した更新候補位置の差分を計算
        Diffx2 = MMx - kx
        # メジャライザーを再計算
        MMtemp = ktemp + np.dot(grad_x, Diffx2) + 0.5 * L * np.linalg.norm(Diffx2, ord=2)**2 

    x = SoftThr(MMx, 1/L)
    return x

# ISTAで値を更新してLASSOの誤差関数で評価。ある程度まで誤差が小さくなったら、解が得られたものとする。
EPSILON = 10**(-5)
y = sinogram.flatten()

err = 1.0
print("err=", err)
# Initial condition
x = np.dot(L.T, y)
print("x.shape", x.shape)
lam = 0.1  # lambda
# P0 = np.linalg.norm(np.dot(L.T, L), ord=2) / lam  # Lipschitz定数の初期値
P0 = 10**6
print("P0", P0)
print("y, A, x", y.shape, L.shape, x.shape)
print("A*x", np.dot(L, x).shape)
print("y-A*x", y - np.dot(L, x))

# ISTAを繰り返す回数を指定
Nt = 100

# 結果とコストを格納する配列
cost = np.zeros(Nt)

for t in range(Nt):
    x = ISTA(L, x, y, lam, P0)
    cost[t] = 0.5 * np.linalg.norm(y - np.dot(L, x), ord=2)**2

reconIm = np.reshape(x, (Nx, Ny))

# 結果を表示
# コスト関数の変化を描画
plt.figure()
plt.plot(np.arange(Nt), cost, marker="o")

# ２次元に戻す
plt.figure( figsize=(8, 8) )
plt.pcolor(reconIm.T)
'''
#---疎行列を使わないバージョン--- end
