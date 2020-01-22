![alt text](https://github.com/Maesimt/introduction-caffe2/blob/master/caffe2.png "Caffe2 - Official logo")

# Guide de démarrage de Caffe2

<a id="table-matieres" />

## Table des matières

<ol>
    <li><a href="#contexte">Contexte</a></li>
    <li><a href="#histoire">L'histoire et le future de Caffe2</a></li>
    <li><a href="#pre-requis">Pré-requis</a></li>
    <li><a href="#on-se-lance">Bon, on se lance !</a></li>
    <li>
        <a href="#configurer-gpu">Configurer un GPU (Optionnel)</a>
        <ol>
            <li><a href="#cuda-toolkit">Nvidia CUDA Toolkit</a></li>
            <li><a href="#CUDDN">Nvidia CUDDN</a></li>
        </ol>
    </li>
        <li>
        <a href="#installation-de-caffe2">Installation de Caffe2</a>
        <ol>
            <li><a href="#installation-de-caffe2">Récupérer le code source</a></li>
            <li><a href="#compiler-code-source">Compiler le code source</a></li>
        </ol>
    </li>
        <li>
        <a href="#verifier-installation">Vérifier l'installation</a>
        <ol>
            <li><a href="#verifier-installation-caffe2">Tester l’installation des librairies de Caffe2.</a></li>
            <li><a href="#verifier-installation-gpu">Tester le build GPU avec Caffe2 (Conditionnel)</a></li>
        </ol>
    </li>
    <li><a href="#installer-interface">Installer un environnement de développement</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#annexe">Annexe</a></li>
</ol>

<a id="contexte" />

## 1. Contexte

Dans le cadre du cours **Mise en place d'un écosystème d'IA** donné dans le programme de spécialisation en intelligence artificielle du Cégep de Sainte-foy à Québec, nous avons à réaliser un travail d'équipe sur un outil utilisé dans le domaine de l'intelligence artificielle. Notre équipe a choisi Caffe2, car nous n'avions pas encore entendu parler de ce framework de Deep Learning jusqu'à présent. Le guide d'installation qui suit est le résultat de l'exploration sur Caffe2 réalisé dans le cadre de la réalisation de ce travail d'équipe.

<a id="histoire" />

## 2. L'histoire de Caffe2

### Les débuts

Caffe est une librairie de deep learning open source. Le projet a été initié par Yangqing Jia pendant qu’il faisait son Phd à Berkeley dans le département BAIR (Berkeley Artificial Intelligence Research). Après avoir ses études à Berkeley, Yanquin a accepté un poste de chercheur chez Google Brain, où il a contribué à TensorFlow et participé à Google LeNet.
Puis, il est aller travailler chez Facebook pour mettre en place leur plateforme générale de AI. Il a initié le mouvement pour créer Caffe2, un des premiers frameworks de réseaux de neurones à offrir des capacités d’IA à haute performance sur l’ensemble des plateformes (Cloud, mobile, systèmes embarqués, etc…).

Qu'est-ce qui a été ajouté dans la version 2:
- Déploiement sur mobile
- Plus de matériel supporté
- Du support pour de l'entraînement distribué.
- Testé en profondeur par Facebook.

### ONNX

Caffe2 c’est cool ça tourne vite et sur beaucoup d'appareils, mais ce n’est pas le framework le plus populaire pour prototyper des idées. Chez facebook avant les débuts de Caffe2, ils avaient beaucoup d’équipe qui développaient dans différents frameworks et tranquillement cette hétérogénéité commençait à devenir plus dure à supporter autant du côté développement qu’aux opérations. Il était nécessaire de connaître beaucoup de frameworks pour développer et les opérations devaient optimiser les environnements de production pour les différentes nuances entre les frameworks. 

Certaines équipes prototypaient avec PyTorch et ensuite ils écrivaient en caffe2 avant de déployer leur modèle en production.
Ils se sont dit que ça serait cool de pouvoir exporter une représentation des modèles entraîner. Et pour ça ils ont creer un format standard ONNX (Open neural network exchange) en partenariat avec AWS et Microsoft. Cela leur a permis de développer leurs idées en Pytorch, et pour aller en production, ils exportaient en ONNX puis ils importaient dans Caffe2 leurs modèles parce que leur plateformes de production étaient mieux optimisée pour Caffe2.

Ils ont fait des connecteurs dans plusieurs  frameworks pour réussir à faire de l'importation et l'exportation de fichiers ONNX. Maintenant, c'est beaucoup plus facile pour les différentes plateformes de supporter un format standardisé qui représente un modèle entraîné que d’optimiser pour chacun des frameworks.

Aujourd'hui, la librairie Caffe2 est incluse dans le projet PyTorch.

<a id="pre-requis" />

## 3. Pré-requis

Vous pouvez suivre les étapes pas à pas sur votre machine si vous roulez la version 18.04 d'Ubuntu. Si vous êtes sur MacOs ou Windows vous pouvez créer une machine virtuelle dans VirtualBox ou VmWare Fusion. Mais si vous voulez utiliser un GPU Nvidia je vous conseille de créer une VM chez un fournisseur Cloud tel que AWS, GCP ou Azure.

Dans tous les cas, assurez-vous d'avoir une machine ou un type d'instance avec au minimum ces spécifications:

- [x] Ubuntu 18.04
- [x] 2 coeurs
- [x] 8 GB de mémoire vive
- [x] 1 GPU Nvidia (Ce guide utilise une Tesla K80)
- [x] Connexion internet

Les guides officiels de Caffe2 recommandent la compilation des sources sur Ubuntu 14.04 et 16.04. Toutefois, il faut tenir compte que ces guides ne sont plus vraiment maintenus depuis que la librairie a été intégrée dans le projet PyTorch. Au 2020-01-05, la version LTS d'Ubuntu est officiellement la 18.04. Pour avoir un maximum de stabilité et de support, ce guide utilise cette version. De plus, les pilotes Nvidia nécessaires pour rouler sur des GPUs y sont aussi davantage maintenus.

<p>:warning: Attention, si vous utilisez la version minimale d'Ubuntu vous pourriez avoir des questions interactives pendant l'installation de certains outils. Si vous voulez éviter ces questions parce que vous voulez intégrer ces commandes dans un Dockerfile. Je conseille de rouler la commande ci-dessous pour définir la région, le clavier, ainsi qu'un ensemble de paramètres qui peuvent être demandés pendant les installations qui vont suivre.</p>

```console
caffe2@demo:~$ sudo update-locale \
    LANG=en_US \
    LANGUAGE=en_US \
    LC_CTYPE="en_US.UTF-8" \
    LC_NUMERIC=en_US.UTF-8 \
    LC_TIME=en_US.UTF-8 \
    LC_COLLATE="en_US.UTF-8" \
    LC_MONETARY=en_US.UTF-8 \
    LC_MESSAGES="en_US.UTF-8" \
    LC_PAPER=en_US.UTF-8 \
    LC_NAME=en_US.UTF-8 \
    LC_ADDRESS=en_US.UTF-8 \
    LC_TELEPHONE=en_US.UTF-8 \
    LC_MEASUREMENT=en_US.UTF-8 \
    LC_IDENTIFICATION=en_US.UTF-8
```

Note: N'oubliez pas d'ajuster le contenu si c'est important pour vous. 

Puis ouvrez un terminal ou une session SSH sur votre VM distante:
```console
caffe2@demo:~$ ssh -i <Fichier clé SSH> <Adresse IP distante>
```

<p align="right">
    <a href="#table-matieres">:scroll: Aller à la table des matières</a>
</p>

<a id="on-se-lance" />

## 4. Bon, on se lance !

Comme à chaque fois, lorsqu'on arrive sur une nouvelle machine, on met à jour la liste des paquets et on fait un "full upgrade" pour récupérer les dernières versions ainsi que leurs dépendances.

```console
caffe2@demo:~$ sudo apt update
caffe2@demo:~$ sudo apt full-upgrade -y
```

Ensuite, on installe un ensemble de paquets qui sont nécessaires dans le reste des étapes de ce guide.

Installer des outils et des librairies.
```console
caffe2@demo:~$ sudo apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libsnappy-dev \
    libprotobuf-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-pip
```

```console
caffe2@demo:~$ sudo apt-get install -y --no-install-recommends \
    libgflags-dev \
    cmake
```

Note : Si utilisez la version minimale d'Ubuntu 18.04 vous devrez peut-être installer Setuptools.

```console
caffe2@demo:~$ pip install -U setuptools
```

<p align="right">
    <a href="#table-matieres">:scroll: Aller à la table des matières</a>
</p>

<a id="configurer-gpu" />

## 5. Configurer un GPU (Optionnel)

Pour accélérer l'entrainement des modèles, je vous conseille de configurer un GPU Nvidia.  
Ça se fait assez bien, simplement suivre ces deux étapes:
1. Installer CUDA Toolkit
2. Installer CUDNN

<a id="cuda-toolkit" />

### 5.1 Nvidia CUDA Toolkit

Pour télécharger le fichier **.deb** de la version CUDA Toolkit 10.0 vous pouvez faire :

```console
caffe2@demo:~$ wget --header="Host: developer.download.nvidia.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,fr;q=0.8" --header="Referer: https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal" --header="Cookie: visid_incap_23871=ETnGeLr3RiO2x7iF3A3KsNAx9F0AAAAAQUIPAAAAAAAhnELzejFRjFyGeF4f+iNk" --header="Connection: keep-alive" "https://developer.download.nvidia.com/compute/cuda/10.0/secure/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb?0f--1Bd4deCHUUDTI5-MDcpFFf21Gkk5PBUB5K8ojnOwNWICkMzOZTAuCKtpiH2Lmj_mN_hboLvFkfK-v-fldLFJPTiKN1TAZwWdxh-CHzfiH4ORhS978XyIlL8H-5jW0nfo4PyqU9p-F0jU5fMbeE3_HhL1F3oglS3UW1pjPcXzSzrdsaONN3Pi-pXMfvKiA8Z890Q-PKITvxuiWv98Zcppym5qF2UOBlND6DU" -O "cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb" -c
```

Pour consulter la liste des versions disponibles, c'est [ici](https://developer.nvidia.com/cuda-toolkit-archive)

Une fois le téléchargement terminé, on peut lancer l'installation :

```console
caffe2@demo:~$ sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
caffe2@demo:~$ sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
caffe2@demo:~$ sudo apt-get update
caffe2@demo:~$ sudo apt-get install cuda -y
```

Après il faut ajouter quelques lignes dans le **bashrc** :

```console
caffe2@demo:~$ echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
caffe2@demo:~$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PAT H}}' >> ~/.bashrc
```

Puis, redémarrer la machine :
```console
caffe2@demo:~$ sudo shutdown -r now
```

Une fois démarrée, vérifier que vous détecter bien votre GPU avec la commande :
```console
caffe2@demo:~$ nvidia-smi
Thu Jan 16 20:21:35 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.104      Driver Version: 410.104      CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
| N/A   71C    P0    90W / 149W |      0MiB / 11441MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

:warning: Si vous avez l'erreur suivante :

```console
caffe2@demo:~$ nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

Vous devrez réinstaller les pilotes graphiques NVidia avec les commandes suivantes:

```console
caffe2@demo:~$ sudo add-apt-repository ppa:graphics-drivers
caffe2@demo:~$ sudo apt-get update
caffe2@demo:~$ sudo apt install nvidia-driver-410 -y
caffe2@demo:~$ sudo shutdown -r now
```

<p align="right">
    <a href="#table-matieres">:scroll: Aller à la table des matières</a>
</p>

<a id="CUDDN" />

### 5.2 Nvidia CUDDN

Récupérer CUDDN à partir des serveurs de Nvidia.

```console
caffe2@demo:~$ wget --header="Host: developer.download.nvidia.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,fr;q=0.8" --header="Referer: https://developer.nvidia.com/rdp/cudnn-download" --header="Cookie: visid_incap_23871=ETnGeLr3RiO2x7iF3A3KsNAx9F0AAAAAQUIPAAAAAAAhnELzejFRjFyGeF4f+iNk" --header="Connection: keep-alive" "https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.0_20191031/cudnn-10.0-linux-x64-v7.6.5.32.tgz?gFSu6QxFwuiM0aTHBfFMq4j4zdqO8GWKDkNYWz7ejwE9hJ4IaRAAdftgV_qql8-DuLiq2m08IgO-OOOqkkOyHooBBGEbFzYDtP4wz5f5POAwEbO0bbn3l4wV23mlvFy21yreAm7MIZ5hmsLpolbFDkgGU-xx4az4iDsjiLWkl8cSJruztjBQobIy0IIpJMWPZ0b1QK6M723U4wQYHOO0xygPGUjkfOCQAQ" -O "cudnn-10.0-linux-x64-v7.6.5.32.tgz" -c
```

Puis exécuter les commandes suivantes pour l'installer :

```console
caffe2@demo:~$ tar -xzvf cudnn-10.0-linux-x64-v7.6.5.32.tgz
caffe2@demo:~$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
caffe2@demo:~$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
caffe2@demo:~$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

<p align="right">
    <a href="#table-matieres">:scroll: Aller à la table des matières</a>
</p>

<a id="installation-de-caffe2" />

## 6. Installation de Caffe2

La documentation officielle de Caffe2 propose plusieurs façons de mettre en place un environnement de développement.

1. Binaires pré-compilés
2. Compiler à partir du code source
3. Image Docker
4. Instance Cloud

L'option 3 et 4 permettent de monter rapidement un environnement de développement avec la dernière release.  
L'option 1 permet d'intégrer les librairies de Caffe2 compilés directement dans un environnement de développement déjà existant.

Dans ce guide, on utilise l'option 2, compiler à partir des sources. C'est intéressant parce que ce n'est pas ce qui est le plus utilisé et c'est nécessaire si vous voulez essayer des versions autres que la dernière release. Ça permet aussi de travailler avec des versions "release-candidate" avant qu'elles soient disponibles ou de travailler avec une version qui contient un correctif.

<a id="recuperer-code-source" />

### 6.1 Récupérer le code source

Pour récupérer la dernière version :

```console
caffe2@demo:~$ git clone https://github.com/pytorch/pytorch.git && cd pytorch
```

Pour récupérer une version précise sur une branche, remplacer **<branchname>** par la branche qui vous intéresse :

```console
caffe2@demo:~$ git --single-branch --branch <branchname> clone https://github.com/pytorch/pytorch.git && cd pytorch
```

<p align="right">
    <a href="#table-matieres">:scroll: Aller à la table des matières</a>
</p>

<a id="compiler-code-source" />

### 6.2 Compiler le code source

Avec la machine décrite dans la section des prérequis, la première compilation des sources a pris environ 6 heures. À la toute fin, les librairies sont déposées dans des répertoires et il se peut que la copie des artefacts échoue si l'utilisateur n'a pas les droits dans les répertoires de sorties. Si ça se produit, donner les droits à l'utilisateur et relancer le setup.py, les fois subséquentes sont beaucoup plus rapides car l'installation utilise un package qui s'appelle Wheel (bdist-wheel) qui empêche la recompilation des modules déjà compilés lors de la dernière installation. L'installation ne fait que rouler les tests si tous les modules ont déjà été compilés mais qu'une erreur est survenue à la toute fin par exemple.

```console
caffe2@demo:~$ git submodule update --init --recursive
caffe2@demo:~$ python setup.py install
```
**Note:** Erreur possible, Pyyaml n’est pas installé. Rouler : `pip install pyyaml`

<p align="right">
    <a href="#table-matieres">:scroll: Aller à la table des matières</a>
</p>

<a id="verifier-installation" />

## 7. Vérifier la compilation des sources

<a id="verifier-installation-caffe2" />

### 7.1 Tester la compilation des librairies de Caffe2

Rouler la commande suivante, vous devriez avoir `Success` si les artefacts de Caffe2 ont bien été compilés et déplacés dans les bons répertoire sur votre VM.

```console
caffe2@demo:~$ cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

Si vous avez des erreurs de droits, surement que votre utilisateur n'a pas les droits dans les dossiers suivants.
```console
caffe2@demo:~$ sudo chown -R $USER /usr/local/lib/python2.7
caffe2@demo:~$ Sudo chown -R $USER /usr/local/bin
```

<a id="verifier-installation-gpu" />

### 7.2 Tester la détection du GPU par Caffe2

Pour tester si Caffe2 est capable d'utiliser votre GPU, on peut voir si le module Workspace de Caffe2.python détecte un GPU Nvidia compatible.
```console
caffe2@demo:~$ cd ~ && python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
1
```
En cas d'échec vous allez avoir ce message.
```console
caffe2@demo:~$ cd ~ && python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())
WARNING:root:This caffe2 python run does not have GPU support. Will run in CPU only mode.
CRITICAL:root:Cannot load caffe2.python. Error: No module named caffe2_pybind11_state
```
:warning: Attention, cette erreur est normale si vous êtes encore dans le répertoire de Pytorch.  
Assurez-vous d'être de retour au home (~)

<a id="installer-interface" />

## 8. Installer un environnement de développement

Pour utiliser Caffe2 facilement, nous allons installer et configurer un Jupyter Notebook sur notre machine virtuelle.
```console
caffe2@demo:~$ pip install jupyterlab
caffe2@demo:~$ pip install notebook
```

Télécharger les tutoriels de Caffe2 afin de pouvoir expérimenter un peu.
```console
caffe2@demo:~$ cd ~ && git clone --recursive https://github.com/caffe2/tutorials caffe2_tutorials
```

On va créer une configuration pour lancer un serveur public ouvert à tous:
```console
caffe2@demo:~$ jupyter notebook --generate-config
caffe2@demo:~$ cd ~/.jupyter
caffe2@demo:~$ rm jupyter_notebook_config.py
caffe2@demo:~$ touch jupyter_notebook_config.py
caffe2@demo:~$ echo "c.NotebookApp.ip = '0.0.0.0'" >> jupyter_notebook_config.py
caffe2@demo:~$ echo "c.NotebookApp.base_url = '/caffe2/'" >> jupyter_notebook_config.py
caffe2@demo:~$ echo "c.NotebookApp.open_browser = False" >> jupyter_notebook_config.py
caffe2@demo:~$ echo "c.NotebookApp.port = 8888" >> jupyter_notebook_config.py
caffe2@demo:~$ echo "c.NotebookApp.notebook_dir = 'caffe2_tutorials'" >> jupyter_notebook_config.py
```

Puis on lance le notebook :
```console
caffe2@demo:~$ cd ~ && jupyter notebook
[I 20:15:26.264 NotebookApp] Serving notebooks from local directory: /home/guillaumecummings/caffe2_tutorials
[I 20:15:26.264 NotebookApp] The Jupyter Notebook is running at:
[I 20:15:26.265 NotebookApp] http://(ubuntu-desktop-2 or 127.0.0.1):8888/caffe2/?token=79807e161b8478910c9bc3c639ced7fbb481556c31ed86b3
[I 20:15:26.265 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 20:15:26.268 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///home/guillaumecummings/.local/share/jupyter/runtime/nbserver-2923-open.html
    Or copy and paste one of these URLs:
        http://(ubuntu-desktop-2 or 127.0.0.1):8888/caffe2/?token=79807e161b8478910c9bc3c639ced7fbb481556c31ed86b3
```
Noter le token après l'exécution de la commande vous en aurez de besoin.

Vérifier que le pare-feu de votre fournisseur cloud laisse bien le trafic sur le port 8888 passé.
Une fois fait, on peut aller sur notre navigateur préféré à l'adresse suivante:
```console
http://<Adresse IP VM>:8888/caffe2?token=79807e161b8478910c9bc3c639ced7fbb481556c31ed86b3
```

Vous devriez voir les tutoriels officiels de Caffe2 qu'on a téléchargé plutôt. 
![alt text](https://github.com/Maesimt/introduction-caffe2/blob/master/jupyter.png "Jupyter web")


<a id="conclusion" />

## 9. Conclusion

### Quoi faire en suite.

Vous avez maintenant une version fonctionnel de Caffe2 et un environnement dans lequel vous pourrez essayer les différents tutoriels. Dans la section références, vous trouverez quelques vidéos intéressant ainsi que les liens vers la documentation officielle. Vous pourriez aussi isoler les installations de python dans des environnements virtuels.

### Le futur

J'ai l'impression que plus ONNX va devenir mature et populaire la nécessité de convertir des modèles vers du Caffe2, pour profiter d'un ensemble de plateformes de déploiement (CPU,GPU,Cloud, Mobile, IOT), va devenir moins nécessaire. Avec de plus en plus plateformes qui vont être optimisées pour rouler des modèles ONNX, je tend à croire que l'utilisation de Caffe2 ne peut que diminuer dans les années à venir.
    
### Références

#### Building an agile AI research-to-production experience - GitHub Universe 2018
Une présentation donnée par le créateur de Caffe2 sur le workflow de AI chez Facebook avant et après la transition de la librairie Caffe2 dans PyTorch.  
Disponible sur https://www.youtube.com/watch?v=dqxaYiVqJFg&t=939s

#### F8: ONNX: Creating A More Open AI Ecosystem
Une conférence organisée par Facebook pour présenter la nouvelle norme ONNX (Open neural network exchange) en partenariat avec AWS et Microsoft.  
Disponible sur https://www.youtube.com/watch?v=Mvnn_Iy29es

#### Documentation Caffe2

[Site officiel](https://caffe2.ai)  
[Documentation](https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=prebuilt)  
[Tutoriels](https://caffe2.ai/docs/tutorials)  
[Caffe2 Github](https://github.com/pytorch/pytorch/tree/master/caffe2)  
[PyTorch Github](https://github.com/pytorch/pytorch)

#### Cours du cégep Sainte-Foy
Cours **Mise en place d'un écosystème d'IA** donné par Mikaël Swawola.

<a id="annexe" />

## 10. Annexe

### Troubleshooting

#### No handlers could be found for logger "caffe2.python.net_drawer"

Si vous avez l'erreur :
```console
No handlers could be found for logger "caffe2.python.net_drawer"
net_drawer will not run correctly. Please install the correct dependencies.
```
Vérifier si vous avez pydot d'installé, si ce n'est pas le cas, rouler : 
```console
caffe2@demo:~$ pip install pydot
```
