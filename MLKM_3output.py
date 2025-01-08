# %%
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
MACHINE = "CPUs" #GPUs,CPUs,GPU #str(sys.argv[1])     روی سی پی یو ران شو
BATCHING = 32 #int(sys.argv[2])                      BATCHING=BATCH_SIZE یعنی 100 تا 100 تا به ماشین بده یعنی 100 اسنپ شات در زمان متفاوت را به صورت تصادفی بده
LENGTH = 100 #int(sys.argv[3])                        تعداد نود ها  
MODEL_NAME = "FCN" #int(sys.argv[4])                           روش اجرای شبکه عصبی (مثل فولی کانکتد باشد یا شبکه عصبی بازگشتی یا ... )
K1 = 0.01
K2 = 0.40 #float(sys.argv[6])                   ابتدای منطقه تست                       ورودی کاربر 
KC1 = 0.60 #float(sys.argv[5])                       J=1 ورودی کاربر          مقدار بحرانی مورد انتظار
K3 = 0.90 #float(sys.argv[7])                   انتهای منطقه تست           
K4 = 2.70
KC2 = 2.95 #float(sys.argv[5])                       J=1 ورودی کاربر          مقدار بحرانی مورد انتظار
K5 = 3.20 #float(sys.argv[7])                   انتهای منطقه تست           
K6 = 3.60



# %% [markdown]
# ### Create a directory file

# %%
PATH = "./"
dir_root = '{path}/saves'
dir_save = '{path}/saves/N{N:d}'
if not os.path.exists(dir_root.format(path = PATH)):
    os.mkdir(dir_root.format(path = PATH))
if not os.path.exists(dir_save.format(path = PATH, N = LENGTH)):
    os.mkdir(dir_save.format(path = PATH, N = LENGTH))

# %% [markdown]
# ### Neural Network

# %%
###############################################################################################################
#######                                                                                                 #######
###############################################################################################################
def ann(length, model_name):
    input = tf.keras.Input(shape = (length,), dtype = tf.float32)
    x = tf.math.cos(input)
    y = tf.math.sin(input)
    z = tf.keras.layers.Concatenate(axis = 1)([x, y])
    
    if model_name == "FCN":                                    # اگر شبکه عصبی فولی کانکتد بود
        z = tf.keras.layers.Dense(128, activation = tf.keras.activations.tanh)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Dense(128, activation = tf.keras.activations.tanh)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Dense(128, activation = tf.keras.activations.tanh)(z)
        z = tf.keras.layers.BatchNormalization()(z)
    output = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax)(z)
    model = tf.keras.Model(input, output)
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5), loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics = ['accuracy'])
    return model


###############################################################################################################
#######                                            CPU/GPU                                              #######
###############################################################################################################
if MACHINE == 'GPUs':
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    BATCH_SIZE_REP = BATCHING
    BATCH_SIZE = BATCH_SIZE_REP * strategy.num_replicas_in_sync
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 4
    def make_model(length, model_name):
        with strategy.scope():
            return ann(length, model_name)
elif MACHINE == 'CPUs' or MACHINE == 'GPU':
    BATCH_SIZE = BATCHING
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 4
    def make_model(length, model_name):
        return ann(length, model_name)
else:   # خطای اجرا نشدن روی هیچ کدام
    #print("Select MACHINE : CPUs/GPUs/GPU")
    raise SystemExit

# %% [markdown]
# #### list of addresses (Phase.txt and label.label)

# %%
###############################################################################################################
#######                                           Phases                                                #######
###############################################################################################################
file_image = 'KMML44000-4000/phase_snapshot/N{N:d}/Phase_snapshot_N{N:d}_J{J:0.2f}.dat'

def list_of_file_image(Kmin, Kmax):
    list = []
    for J in np.arange(Kmin, Kmax + 0.005, 0.01): #مثال آن در پایین 
        list.append(file_image.format(N = LENGTH, J = J))    # به لیست اضافه کن (اون مقادیری رو که )
    return list

###############################################################################################################
#######                                           labels                                                #######
###############################################################################################################
file_label = 'KMML44000-4000/phase_snapshot/Phase_snapshot_J{J:0.2f}.label' #دیرکتوری و اسم فایل که قابل تغیر است (جای آکولاد ها عدد پر میشود) 

def list_of_file_label(Kmin, Kmax):
    list = []
    for J in np.arange(Kmin, Kmax + 0.005, 0.01):     # kmin=2.2 , kmax=2.2 ---> k=2.2
        list.append(file_label.format(J = J))# یک لیست از دو استرینگ بالا تولید میکند و داخل لیست اضافه میکند
    return list # 

# %% [markdown]
# ### Train dataset define

# %%
def make_dataset_train(size_of_ensemble, K1, K2, Kc1, K3, K4, Kc2, K5, K6):
    #Phases list address
    list_of_image = []                                                      
    list_of_image += list_of_file_image(K1, K2)
    list_of_image += list_of_file_image(K3, K4)
    list_of_image += list_of_file_image(K5, K6)
    #print(len(list_of_image))

    #Labels list address
    list_of_label = []                                                      
    list_of_label += list_of_file_label(K1, K2)
    list_of_label += list_of_file_label(K3, K4)
    list_of_label += list_of_file_label(K5, K6)

    number_of_files = len(list_of_label)
    #print(len(list_of_label))

    # Call input data of address and zip them and shuffle them
    labels = tf.data.Dataset.from_tensor_slices(list_of_label).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    images = tf.data.Dataset.from_tensor_slices(list_of_image).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).repeat()

    ###############################################################################################################
    #######                                    Train & Validation                                           #######
    ###############################################################################################################
    # convert string to number(phase and label) & decode Label to (up=[1, 0]/down=[0, 1])
    def decorder(image, label):
        image = tf.strings.split(image)
        image = tf.strings.to_number(image, out_type = tf.float32)
        label = tf.strings.split(label)
        label = tf.strings.to_number(label, out_type = tf.float32)
        def sup(): return tf.constant([0, 0, 1], dtype = tf.int64)
        def mid(): return tf.constant([0, 1, 0], dtype = tf.int64)
        def sub(): return tf.constant([1, 0, 0], dtype = tf.int64)
        # Determine label based on conditions
        is_sup = tf.reduce_all(tf.equal(label, [0, 0, 1]))
        is_mid = tf.reduce_all(tf.equal(label, [0, 1, 0]))
        is_sub = tf.reduce_all(tf.equal(label, [1, 0, 0]))

        label = tf.cond(is_sup, sup, lambda: tf.cond(is_mid, mid, sub))
        return image, label
    
    # Split train and validation
    number_of_total = number_of_files * size_of_ensemble
    number_of_train = int(number_of_total * 0.9)
    number_of_valid = number_of_total - number_of_train
    dataset_train = dataset.skip(number_of_valid)
    dataset_valid = dataset.take(number_of_valid)

    #call def decorder for train and valid
    dataset_train = dataset_train.map(decorder, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset_valid = dataset_valid.map(decorder, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    #Pack and batching train and valid
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    dataset_valid = dataset_valid.batch(BATCH_SIZE, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    
    return dataset_train, dataset_valid, number_of_train, number_of_valid

# %% [markdown]
# ### Test dataset define

# %%
def make_dataset_test(size_of_ensemble,  Kc1, Kc2, Kmin, Kmax):  
    
    #Phases list address
    list_of_image = list_of_file_image(Kmin, Kmax)     # call list_of_file_image to create two str in one list
    list_of_label = list_of_file_label(Kmin, Kmax)    # call list_of_file_label to create two str in one list

    number_of_files = len(list_of_label)                          #پس اندازه آن ۲ است   
    number_of_total = number_of_files * size_of_ensemble          # 2  *   10000   =   20000

    # Call input data of address and zip them and shuffle them
    labels = tf.data.Dataset.from_tensor_slices(list_of_label).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    images = tf.data.Dataset.from_tensor_slices(list_of_image).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset = tf.data.Dataset.zip((images, labels)) # چسباندن ایمج ها و لیبل ها به هم
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).repeat() # بهم ریختن تحول زمانی برای آن که ماشین جهت گیری نکند
    
    ###############################################################################################################
    #######                                    test dataset                                                 #######
    ###############################################################################################################
    # convert string to number(phase and label) & decode Label to (up=[1, 0]/down=[0, 1])
    def decorder(image, label):
        image = tf.strings.split(image)
        image = tf.strings.to_number(image, out_type = tf.float32)
        label = tf.strings.split(label)
        label = tf.strings.to_number(label, out_type = tf.float32)
        def sup(): return tf.constant([0, 0, 1], dtype = tf.int64)
        def mid(): return tf.constant([0, 1, 0], dtype = tf.int64)
        def sub(): return tf.constant([1, 0, 0], dtype = tf.int64)
        # Determine label based on conditions
        is_sup = tf.reduce_all(tf.equal(label, [0, 0, 1]))
        is_mid = tf.reduce_all(tf.equal(label, [0, 1, 0]))
        is_sub = tf.reduce_all(tf.equal(label, [1, 0, 0]))

        label = tf.cond(is_sup, sup, lambda: tf.cond(is_mid, mid, sub))
        return image, label
    
    
    #call def decorder for test dataset
    dataset = dataset.map(decorder, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    #Pack and batching test dataset
    dataset = dataset.batch(BATCH_SIZE, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    
    return dataset, number_of_total

# %% [markdown]
# # Train

# %%
# use CPU/GPU and call ANN model
model = make_model(LENGTH, MODEL_NAME) 
# split train and valid data
dataset_train, dataset_valid, number_of_train, number_of_valid = make_dataset_train(40000, K1, K2, KC1, K3, K4, KC2, K5, K6) 
# train the model
history = model.fit(dataset_train, verbose = 0, epochs = 10, steps_per_epoch = int(number_of_train / BATCH_SIZE), validation_data = dataset_valid, validation_steps = int(number_of_valid / BATCH_SIZE))

# %% [markdown]
# ### plot loss and accuracy

# %%
def plot_history(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train')
    plt.plot(hist['epoch'], hist['val_accuracy'], label = 'Validation')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train')
    plt.plot(hist['epoch'], hist['val_loss'], label = 'Validation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
train_history = '{DIR_SAVE}/Test_{label:s}_history.pdf'
plot_history(history, train_history.format(DIR_SAVE = dir_save.format(path = PATH, N = LENGTH), label = MODEL_NAME))


# %% [markdown]
# ### Save Trained model

# %%
save = '{DIR_SAVE}/Test_{label:s}_save'
if not os.path.exists(save.format(DIR_SAVE = dir_save.format(path = PATH, N = LENGTH), label = MODEL_NAME)):
    os.mkdir(save.format(DIR_SAVE = dir_save.format(path = PATH, N = LENGTH), label = MODEL_NAME))  
model.save(save.format(DIR_SAVE = dir_save.format(path = PATH, N = LENGTH), label = MODEL_NAME))

# %% [markdown]
# # Test Model
# ### Write output of j and probability 

# %%
dat = './Test_data.dat'
f = open(dat, "w")
for i in range(360):           
    J = 0.0100 + 0.01000 * i  
    dataset, number_of_total = make_dataset_test(40000, KC1, KC2, J, J)
    output = tf.reduce_mean(model.predict(dataset, verbose = 0, steps = int(number_of_total / BATCH_SIZE)), 0)

    f.write(str(J)+"\t"+str(output.numpy()[0])+"\t"+str(output.numpy()[1])+"\t"+str(output.numpy()[2])+"\n")

f.close()

# %% [markdown]
# ### plot Phase transition probability

# %%
'''def plot_outputs(outputs, K1, K2, Kc1, K3, K4, Kc2, K5, K6, filename):#تعریف پلاتر
    plt.clf()#پاک کردن صفحه پلات
    plt.xlabel('K')
    plt.ylabel('output')
    plt.axvspan(K1, K2, color = 'C0', alpha = 0.3)      # رسم پس زمینه آبی برای ساب         kcminورودی پلاتر است 
    plt.axvspan(K3, K4, color = 'C3', alpha = 0.3)      # رسم پس زمینه قرمز برای سوپ         kcmaxورودی پلاتر است 
    plt.axvspan(K5, K6, color = 'C7', alpha = 0.3)      # رسم پس زمینه قرمز برای سوپ         kcmaxورودی پلاتر است 

    plt.axvline(x = Kc1, color = 'C1', linestyle = ':')         # خط چین نارنجی نشان دهنده مقدار J=۱
    plt.axvline(x = Kc2, color = 'C1', linestyle = ':')         # خط چین نارنجی نشان دهنده مقدار J=۱
    plt.axvline(x = (K2 + K3) * 0.5, color = 'C3', linestyle = ':')   #خط چین قرمز نشان دهنده مقدار برخورد دو منحنی (میانگین انتهای ساب و ابتدای سوپ)
    plt.axvline(x = (K4 + K5) * 0.5, color = 'C3', linestyle = ':')   #خط چین قرمز نشان دهنده مقدار برخورد دو منحنی (میانگین انتهای ساب و ابتدای سوپ)

    plt.axhline(y = 0.5, color = 'C2', linestyle = ':')        # خط افقی وسط صفحه
    
    plt.plot(outputs[:,0], outputs[:,1], label = 'output 1', color = 'C4') #محور صعودی/ محور ایکس برابر مقادیر کوپلینگ که به ترتیب و فاصله منظم از صفر تا ۲.۲ رفته   و محور وای برابر مقادیر احتمال لیبل اول
    plt.plot(outputs[:,0], outputs[:,2], label = 'output 2', color = 'C5') #محور نزولی  محور ایکس برابر مقادیر کوپلینگ که به ترتیب و فاصله منظم از صفر تا ۲.۲ رفته   و محور وای برابر مقادیر احتمال لیبل دوم
    plt.plot(outputs[:,0], outputs[:,3], label = 'output 3', color = 'C6') #محور نزولی  محور ایکس برابر مقادیر کوپلینگ که به ترتیب و فاصله منظم از صفر تا ۲.۲ رفته   و محور وای برابر مقادیر احتمال لیبل دوم

    plt.legend()                                               # برچسب نوشته های منخنی ها (باکس بالای صفحه)  output1 & output2
    plt.savefig(filename)                                      #ذخیره تصویر تولیدی در آدرس گرفته شده از پلاتر
fig = '{DIR_SAVE}/Test_{label:s}_test.pdf' #آدرس ذخیره فایل و اسم آن

outputs = np.empty((360, 4))       #      پس تعداد گام های کوپلینگ ۲۲۰ عدد است پس گام کوپلینگ ۰.۰۱ است     



plot_outputs(outputs, K1, K2, KC1, K3, K4, KC2, K5, K6, fig.format(DIR_SAVE = dir_save.format(path = PATH, N = LENGTH), label = MODEL_NAME))
plt.savefig('plot.png')'''

# %%



