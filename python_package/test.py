import os
import shutil
current_directory = os.path.dirname(os.path.abspath(__file__))
pretrained_path=current_directory+'/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
file_dir = os.path.expanduser('~/.keras/models')
model_final = file_dir+'/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)
shutil.copyfile(pretrained_path,model_final)
