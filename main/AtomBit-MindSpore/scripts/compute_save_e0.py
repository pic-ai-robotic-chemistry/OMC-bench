import os
import mindspore as ms
import random
import pickle
from tqdm.auto import tqdm

# ==========================================
# 0. 导入你的自定义函数
# ==========================================
# 确保这两个 py 文件在同一目录下
from compute_average_e0 import compute_average_e0
from extxyz_to_pyg_custom import extxyz_to_pyg_custom

def main():
    # ==========================================
    # 1. 配置参数
    # ==========================================
    # 你的数据文件夹
    file_dirs = [r"/home/hxy/005_all", r"/home/hxy/100_all", r"/home/hxy/outcar_selected_xyz", r"/home/hxy/xyz_grouped"]
    
    # 结果保存位置
    SAVE_DIR = "." # 请确保和你的训练脚本一致
    SAVE_NAME = "meta_data.pickle"
    
    CUTOFF = 6.0
    
    # 🎯 只需要收集 3000 ~ 5000 个样本就足够精确了
    TARGET_SAMPLE_COUNT = 30000 

    # ==========================================
    # 2. 搜集并打乱文件
    # ==========================================
    print("🔍 正在搜索文件...")
    all_files = []
    for d in file_dirs:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.xyz')]
            all_files.extend(files)
            print(f"   -> 在 {d} 中找到 {len(files)} 个文件")
            
    if not all_files:
        print("❌ 未找到任何 .xyz 文件！")
        return

    # 打乱文件顺序，确保采样的随机性
    random.shuffle(all_files)
    print(f"📊 总文件数: {len(all_files)}")
    print(f"🎯 目标采样数: {TARGET_SAMPLE_COUNT} (达到即停)")

    # ==========================================
    # 3. 单进程读取 (简单稳定)
    # ==========================================
    collected_samples = []
    
    # 使用 tqdm 显示进度
    pbar = tqdm(all_files, desc="Reading files")
    
    for file_path in pbar:
        # 如果收集够了，直接跳出循环
        if len(collected_samples) >= TARGET_SAMPLE_COUNT:
            break
            
        if os.path.getsize(file_path) == 0:
            continue
            
        try:
            # 读取单个文件
            data_list = extxyz_to_pyg_custom(file_path, cutoff=CUTOFF)
            
            if data_list:
                collected_samples.extend(data_list)
                
            # 更新进度条后缀，显示当前收集进度
            pbar.set_postfix({"Samples": len(collected_samples)})
            
        except Exception as e:
            print(f"⚠️ 跳过错误文件 {os.path.basename(file_path)}: {e}")
            continue

    print(f"\n✅ 采样完成！实际收集样本数: {len(collected_samples)}")

    # ==========================================
    # 4. 计算并保存
    # ==========================================
    if len(collected_samples) > 0:
        print("🧮 正在计算 E0 (这可能需要几秒钟)...")
        
        try:
            # 调用你的计算函数
            e0_dict = compute_average_e0(collected_samples)
            print(f"✅ E0 计算结果: {e0_dict}")
            
            # 确保目录存在
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
                print(f"📂 创建目录: {SAVE_DIR}")
                
            # 保存
            save_path = os.path.join(SAVE_DIR, SAVE_NAME)
            with open(save_path, 'wb') as file:
                pickle.dump({'e0_dict': e0_dict}, file)
            
            print(f"🎉 成功保存到: {save_path}")
            print("💡 下一步：你可以直接运行训练脚本了。")
            
        except Exception as e:
            print(f"❌ 计算过程出错: {e}")
    else:
        print("❌ 没有读取到有效数据，无法计算。")

if __name__ == "__main__":
    main()
