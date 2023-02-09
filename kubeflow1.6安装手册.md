# 简介

Kubeflow 是在 Kubernetes 上运行 TensorFlow、JupyterHub、Seldon 和 PyTorch 等框架的一种很好的方式，Kubeflow很好解决了在kubernetes上运行机器学习负载面临的挑战，全面支撑起用户在机器学习开发、构建、训练与部署这个完整的使用过程。在本文中实现了在自建k8s集群部署和创建Kubeflow 1.6，在Kubeflow中创建并使用Jupter笔记本，并在笔记本中使用TensorFlow在Kubeflow中运行一个简单的模型训练。

# 兼容性

[Amazon EKS and Kubeflow Compatibility | Kubeflow on AWS](https://awslabs.github.io/kubeflow-manifests/docs/about/eks-compatibility/)

|              |               |
| ------------ | ------------- |
| k8s Versions | Kubeflow v1.6 |
| 1.23         | 兼容            |
| 1.22         | 兼容            |
| 1.21         | 不兼容           |
| 1.20         | 不兼容           |

# 依赖

- Kubernetes (up to 1.21) with a default [StorageClass](https://kubernetes.io/docs/concepts/storage/storage-classes/)

- ⚠️ Kubeflow 1.5.0 is not compatible with version 1.22 and onwards. You can track the remaining work for K8s 1.22 support in [kubeflow/kubeflow#6353](https://github.com/kubeflow/kubeflow/issues/6353)

- kustomize (version 3.2.0) ([download link](https://github.com/kubernetes-sigs/kustomize/releases/tag/v3.2.0))

- ⚠️ Kubeflow 1.5.0 is not compatible with the latest versions of of kustomize 4.x. This is due to changes in the order resources are sorted and printed. Please see [kubernetes-sigs/kustomize#3794](https://github.com/kubernetes-sigs/kustomize/issues/3794) and [kubeflow/manifests#1797](https://github.com/kubeflow/manifests/issues/1797). We know this is not ideal and are working with the upstream kustomize team to add support for the latest versions of kustomize as soon as we can.

- kubectl

**故此次我们测试基于以下版本**

| 组件        | 版本    |
| --------- | ----- |
| k8s       | 1.22  |
| kubeflow  | 1.6.0 |
| kustomize | 3.2.0 |

最新版1.6.1尝试多次，始终无法安装

# 安装

```
# 配置科学上网：略
# 设置docker使用代理
root@mgt:~/maojing/image# cat /etc/systemd/system/docker.service.d/http-proxy.conf
[Service]
Environment="HTTP_PROXY=socks5://yourip:9999"
Environment="HTTPS_PROXY=socks5://yourip:9999"
Environment="NO_PROXY=localhost,127.0.0.1,192.168.109.149:8090,10.30.20.233:8090"
# 下载工程
wget https://github.com/kubeflow/manifests/archive/refs/tags/v1.6.0.zip
# 修改配置
root@node01:~/kubeflow/manifests-1.6.0# grep -ri 'dex.auth.svc.cluster.local'
common/oidc-authservice/base/params.env:OIDC_PROVIDER=http://dex.auth.svc.cluster.local:5556/dex
common/dex/overlays/github/config-map.yaml:          redirectURI: http://dex.auth.svc.cluster.local:5556/dex/callback
common/dex/base/config-map.yaml:    issuer: http://dex.auth.svc.cluster.local:5556/dex
# 需要将cluster.local修改为你自己的 cluster name
sed -i 's/cluster.local/ai-test/g' common/oidc-authservice/base/params.env:
sed -i 's/cluster.local/ai-test/g' common/dex/overlays/github/config-map.yaml
sed -i 's/cluster.local/ai-test/g' common/dex/base/config-map.yaml
# 修改tensorboard-controller的配置，新增如下两行解决starting container process caused: chdir to cwd ("/home/nonroot")问题
##
vim apps/tensorboard/tensorboard-controller/upstream/manager/manager.yaml
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext:
        runAsNonRoot: true
        runAsUser: 65532 //新增
        runAsUser: 65532  //新增
      serviceAccount: tensorboard-controller-controller-manager
      serviceAccountName: tensorboard-controller-controller-manager
      terminationGracePeriodSeconds: 10
 
 
# 安装集群
while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
# 检查集群
kubectl get pods -n cert-manager
kubectl get pods -n istio-system
kubectl get pods -n auth
kubectl get pods -n knative-eventing
kubectl get pods -n knative-serving
kubectl get pods -n kubeflow
kubectl get pods -n kubeflow-user-example-com
 
 
# 打开外网访问
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
# 或者你直接使用node port的端口访问也可以
root@node01:~/kubeflow/manifests-1.6.1# kubectl get svc -n istio-system
NAME                    TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                                                                      AGE
authservice             ClusterIP   10.53.17.104    <none>        8080/TCP                                                                     47m
cluster-local-gateway   ClusterIP   10.53.156.130   <none>        15020/TCP,80/TCP                                                             47m
istio-ingressgateway    NodePort    10.53.11.111    <none>        15021:31200/TCP,80:32675/TCP,443:30328/TCP,31400:30936/TCP,15443:32618/TCP   47m
 
 
# 访问地址
http://192.168.123.236:8080 or http://192.168.123.236:32675/
```

# 测试demo

登录[http://192.168.123.236:32675/](http://192.168.123.236:32675/)

创建nodebook

![avatar](https://github.com/maojing0609/install-guide/blob/main/pic/create_notebook.png)


等待容器启动，然后点击CONNECT

![avator](https://github.com/maojing0609/install-guide/blob/main/pic/connect.png)

创建训练集

![avator](https://github.com/maojing0609/install-guide/blob/main/pic/new_py3.png)

demo（需要配置proxy，否则无法拉取google资源）

```
from __future__ import print_function
 
 
import tensorflow as tf
from tensorflow import keras
 
# Helper libraries
import numpy as np
import os
import subprocess
import argparse
 
# Reduce spam logs from s3 client
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
 
def preprocessing():
  fashion_mnist = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
 
  # scale the values to 0.0 to 1.0
  train_images = train_images / 255.0
  test_images = test_images / 255.0
 
  # reshape for feeding into the model
  train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
  test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
 
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 
  print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
  print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))
 
  return train_images, train_labels, test_images, test_labels
 
def train(train_images, train_labels, epochs, model_summary_path):
  if model_summary_path:
    logdir=model_summary_path # + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
 
  model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3,
                        strides=2, activation='relu', name='Conv1'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
  ])
  model.summary()
 
  model.compile(optimizer=tf.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
                )
  if model_summary_path:
    model.fit(train_images, train_labels, epochs=epochs, callbacks=[tensorboard_callback])
  else:
    model.fit(train_images, train_labels, epochs=epochs)
 
  return model
 
def eval(model, test_images, test_labels):
  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('\nTest accuracy: {}'.format(test_acc))
 
def export_model(model, model_export_path):
  version = 1
  export_path = os.path.join(model_export_path, str(version))
 
  tf.saved_model.simple_save(
    keras.backend.get_session(),
    export_path,
    inputs={'input_image': model.input},
    outputs={t.name:t for t in model.outputs})
 
  print('\nSaved model: {}'.format(export_path))
 
 
def main(argv=None):
  os.environ["http_proxy"] = "http://"
  os.environ["https_proxy"] = "http://"
  parser = argparse.ArgumentParser(description='Fashion MNIST Tensorflow Example')
  parser.add_argument('--model_export_path', type=str, help='Model export path')
  parser.add_argument('--model_summary_path', type=str,  help='Model summry files for Tensorboard visualization')
  parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
  args = parser.parse_args(args=[])
 
  train_images, train_labels, test_images, test_labels = preprocessing()
  model = train(train_images, train_labels, args.epochs, args.model_summary_path)
  eval(model, test_images, test_labels)
 
  if args.model_export_path:
    export_model(model, args.model_export_path)
```

执行main方法

![avator](https://github.com/maojing0609/install-guide/blob/main/pic/run.png)



上面是输出结果，输出的前几行显示下载了mnist数据集，下载的训练数据集是60000个图像，测试数据集为10000个图像，并给出了用于训练的超参数，五次Epoch的输出，最后在训练完成后输出了模型的损失值和准确率(*Accuracy*)。



# 参考

[error: unable to recognize &quot;STDIN&quot;: no matches for kind &quot;Profile&quot; in version &quot;kubeflow.org/v1beta1 · Issue #2352 · kubeflow/manifests · GitHub](https://github.com/kubeflow/manifests/issues/2352)

[GitHub - kubeflow/manifests: A repository for Kustomize manifests](https://github.com/kubeflow/manifests/)


