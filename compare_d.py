import numpy as np
import matplotlib.pyplot as plt
import cv2

def align_depth_to_rgb(depth_image, rgb_shape):
    """
    RealSense D435i 参数:
    RGB FOV: 69.4° x 42.5° x 77° (H × V × D)
    Depth FOV: 87° x 58° x 95° (H × V × D)
    """
    # 深度相机和RGB相机的FOV比例
    h_ratio = 69 / 87.0
    v_ratio = 42 / 58.0
    
    # 计算裁剪区域
    h_margin = int((depth_image.shape[1] - depth_image.shape[1] * h_ratio) / 2)
    v_margin = int((depth_image.shape[0] - depth_image.shape[0] * v_ratio) / 2)
    
    # 裁剪深度图像
    cropped_depth = depth_image[v_margin:-v_margin, h_margin:-h_margin]
    
    # 调整到RGB图像大小
    aligned_depth = cv2.resize(cropped_depth, (rgb_shape[1], rgb_shape[0]))
    
    return aligned_depth

def load_and_compare_depths():
    # 加载图像
    rgbd_depth = np.load('input/depth_water-1-3.npy')
    depth_pro = np.load('output/color_water-1-3.npz')['depth']
    rgb_img = cv2.imread('input/color_water-1-3.png')
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # 对齐深度图到RGB视角
    rgbd_depth_aligned = align_depth_to_rgb(rgbd_depth, rgb_img.shape)
    
    # 归一化DepthPro的深度值到合理范围
    # 假设DepthPro输出的是相对深度，需要缩放到实际深度范围
    depth_pro_scaled = depth_pro * np.mean(rgbd_depth_aligned[rgbd_depth_aligned > 0])
    
    plt.figure(figsize=(15, 10))
    
    # RGB图像
    plt.subplot(231)
    plt.imshow(rgb_img)
    plt.title('RGB Image')
    plt.axis('off')
    
    # 对齐后的RGBD深度图
    plt.subplot(232)
    plt.imshow(rgbd_depth_aligned, cmap='viridis')
    plt.colorbar(label='Depth (mm)')
    plt.title('Aligned RGBD Depth')
    plt.axis('off')
    
    # 缩放后的DepthPro深度图
    plt.subplot(233)
    plt.imshow(depth_pro_scaled, cmap='viridis')
    plt.colorbar(label='Depth (mm)')
    plt.title('Scaled DepthPro Depth')
    plt.axis('off')
    
    # 计算对齐后的深度差异
    valid_mask = rgbd_depth_aligned > 0
    depth_diff = np.abs(rgbd_depth_aligned - depth_pro_scaled)
    depth_diff[~valid_mask] = 0
    
    # 显示差异图
    plt.subplot(234)
    plt.imshow(depth_diff, cmap='hot')
    plt.colorbar(label='Depth Difference (mm)')
    plt.title('Depth Difference')
    plt.axis('off')
    
    # 统计信息（只考虑有效深度值）
    valid_rgbd = rgbd_depth_aligned[valid_mask]
    valid_depth_pro = depth_pro_scaled[valid_mask]
    
    rmse, rel_error = calculate_metrics(valid_rgbd, valid_depth_pro, valid_mask)

    print(f"RGBD Depth range: {valid_rgbd.min():.3f} to {valid_rgbd.max():.3f}")
    print(f"Scaled DepthPro range: {valid_depth_pro.min():.3f} to {valid_depth_pro.max():.3f}")
    print(f"Mean absolute difference: {np.mean(depth_diff[valid_mask]):.3f}")
    print(f"Max absolute difference: {np.max(depth_diff[valid_mask]):.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Relative Error: {rel_error:.3f}")
    
    # 深度分布直方图
    plt.subplot(235)
    plt.hist(valid_rgbd.flatten(), bins=50, alpha=0.5, label='RGBD')
    plt.hist(valid_depth_pro.flatten(), bins=50, alpha=0.5, label='DepthPro')
    plt.xlabel('Depth (mm)')
    plt.ylabel('Frequency')
    plt.title('Depth Distribution')
    plt.legend()
    
    plt.tight_layout()
    # Save figure instead of showing it
    plt.savefig('./depth_analysis/depth_comparison.png')
    plt.close()
def calculate_metrics(rgbd, depth_pro, valid_mask):
    # Since rgbd and depth_pro are already masked arrays when passed in
    # we don't need to apply the mask again
    rmse = np.sqrt(np.mean((rgbd - depth_pro)**2))
    rel_error = np.mean(np.abs(rgbd - depth_pro) / rgbd)
    
    # Add error checking
    if np.isnan(rmse) or np.isnan(rel_error):
        print("Warning: NaN values detected in metrics calculation")
        print(f"RGBD range: {rgbd.min():.3f} to {rgbd.max():.3f}")
        print(f"DepthPro range: {depth_pro.min():.3f} to {depth_pro.max():.3f}")
        
    return rmse, rel_error

if __name__ == "__main__":
    load_and_compare_depths()
