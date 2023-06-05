# mvsformer

```
参考
https://github.com/JeffWang987/MVSTER
https://github.com/megvii-research/TransMVSNet
https://github.com/drinkingcoder/FlowFormer-Official
```

## 1 总体结构

对于训练核心的代码有如下几个：

- `train.py`: 整体深度学习框架(参数处理、dataset和DataLoader构建、epoch batch训练、计算loss梯度下降、读取/保存模型等)
- `models`
    - `module.py`: mvsnet所需的网络基础架构和方法(网络组成模块、投影变换homo_wraping、深度回归depth_regression)
    - `mvsnet.py`: MVSNet整体Pipeline(特征提取 深度回归 残差优化网络定义、mvsnet_loss定义、核心四大步骤: 特征提取，cost volume构建、代价体正则化、深度图refine)
- `datasets`
    - `data_yao.py`: 定义MVSDataset(ref图和src图，投影矩阵，深度图真值，深度假设列表，mask)
- `utils.py`: 一些小工具(logger、不同度量指标、系列wrapper方法)

项目整体文件结构

- `checkpoints`(自己创建): 保存训练好的模型和tensorboard数据可视化所需的数据
- `outputs`(自己创建): test的时候输出的预测深度图和点云融合后的点云文件等
- `lists`: train, valid, test用的scan选择列表
- `evaluations`: dtu数据集官方提供的matlab代码，主要用于测试重建点云的质量

---

## 2 DTU数据集结构

共128个scan

- train: 79个
- val: 18个
- test: 22个

### Train

【Cameras】

- `pair.txt`: 只有一个，每个scan通用的
    - 每个场景49个view的配对方式
    
    ```
    49   # 场景的总视点数
    
    0    # ref视点
    src视点总数 第十个视点 视点选取时匹配的score   第一个视点
    10           10          2346.41             1       2036.53 9 1243.89 12 1052.87 11 1000.84 13 703.583 2 604.456 8 439.759 14 327.419 27 249.278 
    
    1
    10 9 2850.87 10 2583.94 2 2105.59 0 2052.84 8 1868.24 13 1184.23 14 1017.51 12 961.966 7 670.208 15 657.218 
    
    2
    10 8 2501.24 1 2106.88 7 1856.5 9 1782.34 3 1141.77 15 1061.76 14 815.457 16 762.153 6 709.789 10 699.921
    ```
    
- `train/xxxxx_cam.txt`：49个 ，每个视点有一个相机参数，不同scan是一致的(与Camera根目录下的camera参数文件不一样，代码里用的是train这个)
    
    - 相机外参、相机内参、最小深度、深度假设间隔(之后还要乘以interval_scale才送去用)

【Depths】

 深度图 & 深度图可视化

- 共128个scan
- `depth_map_00xx.pfm`: 每个scan文件夹里49个视角的深度图  (深度以mm为单位)
- ****`depth_visual_00xx.png`: 还有49张深度图的png版本被用作**mask**(二值图，值为1的像素是深度可靠点，后续训练时只计算这些点的loss)

【Rectified】

原图

- 共128个scan
- 每个scan文件夹里里共49个视角*7种光照 = 343张图片
- 命名：`rect_[view]_[light]_r5000.png`
- 图片尺寸：640*512

### Test

共有22个基准测试场景，对于每一个scan文件夹

- `pair.txt`: 49个场景的配对信息，与train/Cameras/pair.txt是一样的，只是在每个scan里都复制了一份
- `images/`: 该场景下49张不同视角的原始图片
- `cams/`: 每个视点下的相机参数文件(❓不知道为什么有64个)

---

## 3 具体模块

### 代码中的数据维度

- `B`: batch size 在研究数据维度时可以直接将这维去掉
- `C`: 图像特征维度 最开始是3-channels，后来通过特征提取网络变成32维
- `Ndepth`: 深度假设维度，这里是192个不同的深度假设
- `H`: 图像高度，原始是640
- `W`: 图像宽度，512

> 注：在后文维度中最后的H和W可能相反，只为了简单理解并不代表实际运行
>

### dtu_yao/MVSDataset

- `MVSDataset(datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06)`
    - `datapath`: 数据集路径
    - `listfile`: 数据列表(用哪些scan训练和测试都是提前定好的)
    - `mode`: train or test
    - `nviews`: 多视点总数(实现中取5=1ref+4src)
    - `ndepths`: 深度假设数(默认假设192种不同的深度)
    - `interval_scale`: 深度间隔缩放因子(数据集文件中定义了深度采样间隔是2.5，再把这个值乘以缩放因子，最终每隔2.5*1.06取一个不同的深度假设)
- `build_list()`: 构建训练样本条目，最终的`meta`数组中共用27097条数据，每个元素如下：
  
    ```python
    # scan   light_idx      ref_view          src_view
    # 场景    光照(0~6)  中心视点(估计它的深度)    参考视点
    ('scan2', 0, 0, [10, 1, 9, 12, 11, 13, 2, 8, 14, 27])
    ```
    
    - 79个不同的scan
    - 7种不同的光照
    - 每个scan有49个不同的中心视点
- `read_img()`: 将图像归一化到0～1(神经网络训练常用技巧，激活函数的取值范围大都是0～1，便于高效计算)
- `read_cam_file()`: 相机外参、相机内参、最小深度(都为425)、深度假设间隔(都为2.5)
- `getitem()`: 取一组用来训练的数据
    - `imgs`: 1ref + 4src（都归一化到0-1） (3, 3, 512, 640) 3个3channel的512*640大小的图片
    - `proj_metrices`: 3个4*4投影矩阵$\begin{bmatrix} R_{3,3} \ t_{3,1} \\ 0 \ 1 \end{bmatrix}$  (3, 4, 4)
        - 这里是一个视点就有一个投影矩阵，因为MVSNet中所有的投影矩阵都是相对于一个基准视点的投影关系，所以如果想建立两个视点的关系，他们两个都有投影矩阵，可以大致理解为 $B = P_B^{-1}P_AA$
        - 投影矩阵按理说应该是3*3的，这里在最后一行补了[0, 0, 0, 1]为了后续方便计算，所以这里投影矩阵维度是4*4
    - `depth`: ref的深度图 (512, 640)
    - `depth_values`: ref将来要假设的所有深度值 (从425开始每隔2.5取一个数，一共取192个)
        - 2.5还要乘以深度间隔缩放因子
    - `mask`: ref深度图的mask(0-1二值图)，用来选取真值可靠的点(512, 640)

### dtu_yao_eval.py/MVSDataset

- 参数与训练时完全一致
- `build_list`: 构建视点匹配列表，最终meta长度为1078，每个元素如下，与train相比没有光照变化
  
    ```python
    ('scan1', 0, [10, 1, 9, 12, 11, 13, 2, 8, 14, 27])
    ```
    
- `read_cam_file()`: 内参除4，最终生成的深度图也下采样4倍
- `read_img()`: 裁掉下方的16个像素，图像尺寸变为1184*1600，裁剪后不需要修改内存
- `getitem()`:
    - `imgs`: (5, 3, 1184, 1600) 测试的时候有5张图像，读的时候每张被裁剪掉了下面16像素
    - `proj_metrics`: 5个投影矩阵，注意内参除了4倍
    - `depth_values`: 深度假设范围，仍然是从425开始每隔2.5取一个数，一共192个
    - `filename`: ref所在的文件夹名，如`scan1/`



### eval.py

- 相机参数读取是内参intrinsics要除4
- 测试时参考图像用了5个视点
1. 生成所有测试图片的深度图和confidence图
2. 通过光度一致性和几何一致性优化深度图

`save_depth()`: 通过MVSNet进行test生成深度图的核心步骤

- 首先构建MVSDataset和Loader
- 对于每一条训练数据通过模型
    - 输入：1ref + 4src，每个视点的投影矩阵，深度假设list
    - 输出：深度图，photometric confidence
        - 深度图里的数据都是668.08545, 559.7229这类的真实物理距离(不满足像素的取值所以在mac上直接看是一片空白的)
        - 置信度里的数据是0～1之间的小数
- 将模型输出的两张图分别保存成pfm

<img src="https://upload-images.jianshu.io/upload_images/12014150-fb67235f36d4322b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" alt="Untitled 1.png" width="40%;" />

`reproject_with_depth()`: 将ref的点投影到src上，再投影回来

- 参数：ref的深度图和内外参，src的深度图和内外参
- 返回值：重投影回来的深度图，重投影回来的x和y坐标，在src上的x和y坐标 尺寸都是(128, 160)

`check_gemoetric_consistency()`: 几何一致性检验，调用上面的方法进行重投影，重投影后像素偏移<1 && 深度差<1%则通过校验

- 参数：ref的深度图和内外参，src的深度图和内外参
- 返回值：
    - mask: 通过几何检验的mask图
    - depth_reprojected: 重投影后的深度图
    - x2d_src： ref这些像素在src上的坐标
    - y2d_src： ref这些像素在src上的坐标

`filter_depth()`: 通过光度一致性约束和几何一致性约束filter上一步得到的深度图

- `photo_mask`: 置信度图>0.8
- `geometric_mask`: 至少3个src满足上面的几何一致性校验(重投影后像素偏移<1 && 深度差<1%)
- filter每张ref的x y depth，并赋予颜色
- 最终融合生成最后的点云

