进入虚拟环境Image_Clas_pytorch（使用自己的虚拟环境或者Docker即可）
source activate Image_Clas_pytorch


#### train.py
（1）进入工程文件所在根目录
（2）修改train.py中关于模型、图像增强方式、数据集路径、batch size、预训练模型路径、优化器和学习率、epoch、模型保存路径等信息
（3）python train.py

#### pytorch2onnx
（1）修改pytorch2onnx文件夹中pytorch2onnx_test.py文件中关于img_path、input_model_path、out_onnx_model_path、input_shape等信息
（2）python pytorch2onnx_test.py


#### onnx-sim
（1）打开终端，进入onnx-sim虚拟环境。（进入自己的onnx-sim环境）
     source activate onnx-sim
（2）执行onnx-sim
     python -m onnxsim 【onnx源文件所在路径】 【onnx-sim文件保存路径】 --input-shape 1,3,224,224
