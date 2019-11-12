# coding=utf-8
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import math
import numpy as np
import os
import argparse
# import matplotlib
import imghdr
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import pickle as pkl
import datetime
import shutil
from random_eraser import get_random_eraser
current_directory = os.path.dirname(os.path.abspath(__file__))
pretrained_path=current_directory+'/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
file_dir = os.path.expanduser('~/.keras/models')
model_final = file_dir+'/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)
shutil.copyfile(pretrained_path,model_final)
parser = argparse.ArgumentParser()
# parser.add_argument('dataset_root')
# parser.add_argument('classes')
# parser.add_argument('result_root')
parser.add_argument('--epochs_pre', type=int, default=10)
parser.add_argument('--epochs_fine', type=int, default=4)
parser.add_argument('--batch_size_pre', type=int, default=16)
parser.add_argument('--batch_size_fine', type=int, default=16)
parser.add_argument('--lr_pre', type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=1e-4)
parser.add_argument('--snapshot_period_pre', type=int, default=1)
parser.add_argument('--snapshot_period_fine', type=int, default=1)
# parser.add_argument('--split', type=float, default=0.8)

eraser = get_random_eraser()

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def generate_from_paths_and_labels(input_paths, labels, batch_size, input_size=(329,329),aug=False,mixup=True):
    num_samples = (len(input_paths)//batch_size)*batch_size
    while 1:
        perm = np.random.permutation(num_samples)
        input_paths = input_paths[perm]
        labels = labels[perm]
        for i in range(0, num_samples, batch_size):
            inputs = list(map(
                lambda x: image.load_img(x, target_size=input_size),
                input_paths[i:i+batch_size]
            ))
            inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                inputs
            )))
            if aug:
                # random horizontal flip
                # axis: 0 row ,1 col, 2 channel
                inputs = np.array(list(map(
                    lambda x: flip_axis(x, 1) if np.random.random() < 0.5 else x,
                    inputs
                )))
                # print(inputs.shape)
                #random rotation
                # inputs = np.array(list(map(
                #     lambda x: image.random_rotation(x,30,row_axis=0, col_axis=1, channel_axis=2),
                #     inputs
                # )))
                # #random shift
                # inputs = np.array(list(map(
                #     lambda x: image.random_shift(x,0.1,0.1),
                #     inputs
                # )))
                #cut out
                inputs = np.array(list(map(
                    lambda x: eraser(x),
                    inputs
                )))
            inputs = preprocess_input(inputs)
            # print(inputs, labels[i:i+batch_size])
            if mixup:
                y = np.array(labels[i:i+batch_size])
                # print y
                l = np.random.beta(0.2, 0.2, batch_size//2)
                X_l = l.reshape(batch_size//2, 1, 1, 1)
                y_l = l.reshape(batch_size//2, 1)

                X1 = inputs[:batch_size//2]
                X2 = inputs[batch_size//2:]
                X = X1 * X_l + X2 * (1 - X_l)
                y1 = y[:batch_size//2]
                y2 = y[batch_size//2:]
                y_new = (y1 * y_l + y2 * (1 - y_l))
                # print y_new
                yield (X, y_new)
            else:
                yield (inputs, labels[i:i+batch_size])

def main(args):

    # ====================================================
    # Preparation
    # ====================================================
    # parameters
    epochs = args.epochs_pre + args.epochs_fine
    args.dataset_root = os.environ['IMAGE_TRAIN_INPUT_PATH']
    args.result_root = os.environ['MODEL_INFERENCE_PATH']
    class_dict ={'奶粉': 26, '纸箱': 75, '胶水': 79, '吹风机': 16, '塑料玩具': 20, '椅子': 53, '充电器': 7, '塑料袋': 23, '纸尿裤': 73, '牙刷': 62, '剃须刀': 11, '辣椒': 90, '土豆': 17, '瓶盖': 66, '一次性塑料手套': 1, '抹布': 38, '杏核': 49, '充电线': 10, '塑料盖子': 22, '干电池': 28, '烟盒': 61, '中性笔': 4, '旧镜子': 46, '充电宝': 8, '鼠标': 99, '水彩笔': 55, '蒜皮': 85, '旧玩偶': 45, '退热贴': 92, '废弃食用油': 30, '青椒': 96, '口服液瓶': 15, '一次性纸杯': 3, '纽扣': 76, '指甲油瓶子': 40, '插座': 42, '充电电池': 9, '塑料桶': 19, '袜子': 89, '电视机': 67, '护手霜': 35, '手表': 32, '红豆': 72, '衣架': 88, '消毒液瓶': 60, '医用棉签': 14, '扫把': 34, '海绵': 59, '塑料包装': 18, '菜刀': 81, '蛋_蛋壳': 87, '剪刀': 12, '暖宝宝贴': 47, '纸巾_卷纸_抽纸': 74, '糖果': 71, '铅笔屑': 94, '头饰': 25, '泡沫盒子': 57, '打火机': 33, '杀虫剂': 48, '毛毯': 54, '自行车': 80, '耳机': 77, '信封': 6, '酸奶盒': 93, '作业本': 5, '拖把': 39, '外卖餐盒': 24, '水龙头': 56, '旧帽子': 44, '蒜头': 84, '白糖_盐': 69, '蚊香': 86, '快递盒': 31, '胶带': 78, '菜板': 82, '抱枕': 37, '洗面奶瓶': 58, '空调机': 70, '废弃衣服': 29, '面膜': 97, '香烟': 98, '无纺布手提袋': 43, 'PET塑料瓶': 0, '姜': 27, '护肤品玻璃罐': 36, '过期化妆品': 91, '陶瓷碗碟': 95, '化妆品瓶': 13, '棉签': 52, '指甲钳': 41, '牛奶盒': 65, '牙签': 63, '塑料盆': 21, '葡萄干': 83, '果皮': 51, '牙膏皮': 64, '一次性筷子': 2, '电风扇': 68, '杯子': 50}
    # load class names
    num_classes = 100
    # print classes[87]
    # make input_paths and labels
    input_paths, labels = [], []
    for class_name in os.listdir(args.dataset_root):
        # print(class_name)
        class_root = os.path.join(args.dataset_root, class_name)
        # print class_root
        class_id = class_dict[class_name]
        # print class_root,class_id
        for path in os.listdir(class_root):
            path = os.path.join(class_root, path)
            if imghdr.what(path) == None:
                # this is not an image file
                continue
            input_paths.append(path)
            labels.append(class_id)

    # convert to one-hot-vector format
    labels = to_categorical(labels, num_classes=num_classes)

    # convert to numpy array
    input_paths = np.array(input_paths)

    # shuffle dataset
    perm = np.random.permutation(len(input_paths))
    labels = labels[perm]
    input_paths = input_paths[perm]

    # split dataset for training and validation
    # border = int(len(input_paths) * args.split)
    train_labels = labels[:]
    train_input_paths = input_paths[:]
    print("Training on %d images and labels" % (len(train_input_paths)))
    # print("Validation on %d images and labels" % (len(val_input_paths)))

    # create a directory where results will be saved (if necessary)
    if os.path.exists(args.result_root) == False:
        os.makedirs(args.result_root)

    # ====================================================
    # Build a custom Xception
    # ====================================================
    # instantiate pre-trained Xception model
    # the default input shape is (299, 299, 3)
    # NOTE: the top classifier is not included
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(329,329,3))

    # create a custom top classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    # ====================================================
    # Train only the top classifier
    # ====================================================
    # freeze the body layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(lr=args.lr_pre),
        metrics=['accuracy']
    )

    # train
    hist_pre = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=args.batch_size_pre,
            aug=True
        ),
        steps_per_epoch=math.ceil(len(train_input_paths) / args.batch_size_pre),
        epochs=args.epochs_pre,
        # validation_data=generate_from_paths_and_labels(
        #     input_paths=val_input_paths,
        #     labels=val_labels,
        #     batch_size=args.batch_size_pre
        # ),
        # validation_steps=math.ceil(len(val_input_paths) / args.batch_size_pre),
        # verbose=1,
        # callbacks=[
        #     ModelCheckpoint(
        #         filepath=os.path.join(args.result_root, 'model_pre_ep{epoch}_valloss{val_loss:.3f}.h5'),
        #         period=args.snapshot_period_pre,
        #     ),
        # ],
    )
    # model.save(os.path.join(args.result_root, 'model_pre_final.h5'))

    # ====================================================
    # Train the whole model
    # ====================================================
    # set all the layers to be trainable
    for layer in model.layers:
        layer.trainable = True

    # recompile
    model.compile(
        optimizer=Adam(lr=args.lr_fine),
        loss=categorical_crossentropy,
        metrics=['accuracy'])

    # train
    hist_fine = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=args.batch_size_fine,
            aug=False
        ),
        steps_per_epoch=math.ceil(len(train_input_paths) / args.batch_size_fine),
        epochs=args.epochs_fine,
        # validation_data=generate_from_paths_and_labels(
        #     input_paths=val_input_paths,
        #     labels=val_labels,
        #     batch_size=args.batch_size_fine
        # ),
        # validation_steps=math.ceil(len(val_input_paths) / args.batch_size_fine),
        # verbose=1,
        # callbacks=[
        #     ModelCheckpoint(
        #         filepath=os.path.join(args.result_root, 'model_fine_ep{epoch}_valloss{val_loss:.3f}.h5'),
        #         period=args.snapshot_period_fine,
        #     ),
        # ],
    )
    MODEL_PATH = os.path.join(args.result_root)
    # model.save(os.path.join(args.result_root, 'model_fine_final.h5'))
    tf.keras.experimental.export_saved_model(model, MODEL_PATH + '/SavedModel')
    # ====================================================
    # Create & save result graphs
    # ====================================================
    # concatinate plot data
    # acc = hist_pre.history['acc']
    # val_acc = hist_pre.history['val_acc']
    # loss = hist_pre.history['loss']
    # val_loss = hist_pre.history['val_loss']
    # acc.extend(hist_fine.history['acc'])
    # val_acc.extend(hist_fine.history['val_acc'])
    # loss.extend(hist_fine.history['loss'])
    # val_loss.extend(hist_fine.history['val_loss'])

    # # save graph image
    # plt.plot(range(epochs), acc, marker='.', label='acc')
    # plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.xlabel('epoch')
    # plt.ylabel('acc')
    # plt.savefig(os.path.join(args.result_root, 'acc.png'))
    # plt.clf()

    # plt.plot(range(epochs), loss, marker='.', label='loss')
    # plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.savefig(os.path.join(args.result_root, 'loss.png'))
    # plt.clf()

    # # save plot data as pickle file
    # plot = {
    #     'acc': acc,
    #     'val_acc': val_acc,
    #     'loss': loss,
    #     'val_loss': val_loss,
    # }
    # with open(os.path.join(args.result_root, 'plot.dump'), 'wb') as f:
    #     pkl.dump(plot, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
