# Projeto de Treinamento de Redes Neurais Yolo

## Introdução

Bem-vindo ao projeto de treinamento de redes neurais YOLOv5 e YOLOv8. Este repositório contém todo o código e documentação necessários para reproduzir o treinamento personalizado de modelos YOLO utilizando a plataforma Roboflow. O objetivo desta documentação é guiar novos pesquisadores para que eles possam replicar nosso trabalho de maneira rápida e eficiente.

## Pré-requisitos

Antes de começar, certifique-se de ter os seguintes itens instalados:

- Python 3.8 ou superior
- Git
- Conta no Google Colab
- Conta na plataforma Roboflow

## Instruções de Instalação

Siga os passos abaixo para configurar o ambiente de desenvolvimento:

# Projeto de Treinamento de Redes Neurais YOLOv5 e YOLOv8

## Introdução

Bem-vindo ao projeto de treinamento de redes neurais YOLOv5 e YOLOv8. Este repositório contém todo o código e documentação necessários para reproduzir o treinamento personalizado de modelos YOLO utilizando a plataforma Roboflow. O objetivo desta documentação é guiar novos pesquisadores para que eles possam replicar nosso trabalho de maneira rápida e eficiente.

## Pré-requisitos

Antes de começar, certifique-se de ter os seguintes itens instalados:

- Python 3.8 ou superior
- Git
- Conta no Google Colab
- Conta na plataforma Roboflow

## Instruções de Instalação

Siga os passos abaixo para configurar o ambiente de desenvolvimento:

1. Clone este repositório:
    ```sh
    git clone https://github.com/eilucasbar/Perception-camaro.git
    cd Percepcao-camaro
    ```

2. Crie um ambiente virtual:
    ```sh
    python -m venv env
    source env/bin/activate  # No Windows use `env\Scripts\activate`
    			      
    ```

## Uso

Para executar o projeto, siga os passos abaixo:

1. **Prepare os Dados:**
    - Acesse a plataforma Roboflow e prepare seu conjunto de dados rotulando as imagens necessarias para realizar o treinamento, existe uma média de 600 - 1000 imagens para cada segmento. Elas devem ser coletadas em diferentes condições de iluminação, clima, ângulos, e cenários para garantir a robustez do modelo.
Deve-se garantir que cubram a variação de tamanho, orientação e oclusão dos objetos. 
    - Siga as instruções para gerar um link de exportação para o formato YOLOv5 ou YOLOv8.
    - Você usará proprio notebook do roboflow para realizar este treinamento, no caso de duvida pode acessar dentro da plataforma o tutorial em video de como fazer este treinamento.
    

2. **Treinamento no Google Colab:**
    - Acesse os notebooks específicos para treinamento no Google Colab:
      - [Treinamento YOLOv5](https://colab.research.google.com/github/roboflow/notebooks/blob/main/notebooks/train-yolov5-object-detection-on-custom-data.ipynb)
      - [Treinamento YOLOv8](https://colab.research.google.com/github/roboflow/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb)

    - Siga as instruções no notebook para configurar e iniciar o treinamento.
    - Utilize o link de exportação gerado pelo Roboflow para carregar os dados.
    - Para fazer o downloado da pasta de treinamento da sua rede neural depois de ter seguido todo o passo a passo, você deve utilizar o seguinte código no google Collab (utilizando como referencia o treinamento de uma yoloV5):
```
from google.colab import files
import shutil
Compactar a pasta "yolov5" em um arquivo zip
shutil.make_archive('/content/yolov5', 'zip', '/content/yolov5')
Baixar o arquivo zip para a sua máquina local
files.download('/content/yolov5.zip')
```


3. **Executar o Notebook:**
    - No Google Colab, siga as instruções no notebook para configurar o ambiente e executar o treinamento.
    - Certifique-se de ajustar os hiperparâmetros conforme necessário para seu conjunto de dados.

## Estrutura do Projeto

- `data/` : Contém os dados brutos e processados
- `notebooks/` : Contém os notebooks 
  - `yolov5_training.ipynb` : Notebook para treinamento do YOLOv5
  - `yolov8_training.ipynb` : Notebook para treinamento do YOLOv8
- `tests/` : Contém os testes realizados
- `README.md` : Este arquivo de documentação
- `requirements.txt` : Lista de dependências do projeto

### Tutorial: Implementação de Detecção de Zonas Proxêmicas com YOLOv5

Este tutorial irá guiá-lo desde a configuração inicial até a implementação completa de um sistema de detecção de zonas proxêmicas usando YOLOv5, incluindo a classificação e armazenamento de imagens em pastas e a detecção em tempo real com a webcam.

#### Estrutura do Projeto

```plaintext
proxemic-detection/
├── images/
│   ├── classified/
│   └── unclassified/
├── scripts/
│   ├── classify_images.py
│   └── detect_webcam.py
├── yolov5/
│   └── ... (YOLOv5 repository files)
├── requirements.txt

```

### 1. **Preparação do Ambiente**

#### 1.1. **Clone o Repositório YOLOv5**

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
```

#### 1.2. **Instale as Dependências**

Crie um ambiente virtual Python e instale as dependências necessárias:

```bash
python -m venv yolov5-env
source yolov5-env/bin/activate  # Linux
yolov5-env\Scripts\activate  # Windows

pip install -r requirements.txt
pip install opencv-python-headless  # Adicional para OpenCV
```

### 2. **Classificação e Armazenamento de Imagens**

Crie um script para classificar imagens e armazená-las em uma pasta específica.

#### 2.1. **Script de Classificação de Imagens**

Crie o arquivo `classify_images.py` dentro da pasta `scripts`:

```python
import os
import cv2
import torch
import shutil

# Caminhos
model_path = 'C:/Users/Lucas/Teste ZP/treinamento/yolov5/runs/train/yolov5x_results/weights/best.pt'
image_folder = 'C:/Users/lucas/Teste ZP/treinamento/yolov5/Detecção-camaro-7/test/images'
classified_folder = 'C:/Users/lucas/Teste ZP/treinamento/yolov5/Detecção-camaro-7/classified'

# Verificar se a GPU está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar o modelo YOLOv5 treinado
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.to(device)

# Dicionário de classes
class_labels = {
    0: 'DynamicsLifeForm',
    1: 'DynamicsObjects',
    2: 'StaticLifeForm',
    3: 'StaticObject'
}

# Criar pastas de classificação se não existirem
for label in class_labels.values():
    os.makedirs(os.path.join(classified_folder, label), exist_ok=True)

# Classificar e mover imagens
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    
    if image is None:
        continue

    # Preprocessar a imagem
    img_tensor = torch.from_numpy(image).float().to(device) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # Fazer a detecção
    results = model(img_tensor)

    # Verificar a classe mais confiável
    detections = results.xyxy[0].cpu().numpy()
    if len(detections) > 0:
        cls = int(detections[0][5])
        label = class_labels.get(cls, 'Unknown')
        dest_path = os.path.join(classified_folder, label, filename)
        shutil.move(image_path, dest_path)
        print(f'Movido {filename} para {label}')

print('Classificação concluída!')
```

### 3. **Detecção em Tempo Real com a Webcam e com a ZED 2i**

Crie um script para realizar a detecção em tempo real usando a camea conectada no computador e desenhar as zonas proxêmicas.

#### 3.1. **Script de Detecção em Tempo Real com webcam**

Crie o arquivo `detect_webcam.py` dentro da pasta `scripts`:

```python
import cv2
import torch

# Caminho para o modelo treinado
model_path = 'C:/Users/Lucas/Teste ZP/treinamento/yolov5/runs/train/yolov5x_results/weights/best.pt'

# Verificar se uma GPU está disponível e configurar o dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar o modelo YOLOv5 treinado
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.to(device)  # Mover o modelo para a GPU se disponível

# Dicionário de classes
class_labels = {
    0: 'DynamicsLifeForm',
    1: 'DynamicsObjects',
    2: 'StaticLifeForm',
    3: 'StaticObject'
}

class_colors = {
    'DynamicsLifeForm': (0, 255, 0),  # Verde
    'DynamicsObjects': (0, 0, 255),   # Vermelho
    'StaticLifeForm': (255, 0, 0),    # Azul
    'StaticObject': (255, 255, 0)     # Amarelo
}

zone_radius = {
    'DynamicsLifeForm': 60,  # Zona maior
    'DynamicsObjects': 30,   # Zona menor
    'StaticLifeForm': 0,     # Sem zona específica
    'StaticObject': 0        # Sem zona específica
}

def draw_boxes_and_zones(image, results):
    detections = results.xyxy[0].cpu().numpy()  # Resultados da detecção

    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(cls)
        confidence = conf
        label = class_labels.get(class_id, 'Unknown')
        color = class_colors.get(label, (255, 255, 255))  # Branco como cor padrão
        radius = zone_radius.get(label, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f'{label}: {confidence:.2f}'
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if radius > 0:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(image, (center_x, center_y), radius, color, 1)

    return image

def process_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame da webcam.")
            break

        img_tensor = torch.from_numpy(frame).float().to(device) / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        results = model(img_tensor)

        frame = frame[..., ::-1]
        frame_with_boxes_and_zones = draw_boxes_and_zones(frame, results)

        cv2.imshow('YOLOv5 Detection', frame_with_boxes_and_zones)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_webcam()
```
#### 3.2. **Script de Detecção em Tempo Real Utilzando a ZED 2i**

Crie o arquivo `detect_zed.py` dentro da pasta `scripts`:
```python
import cv2
import torch
import pyzed.sl as sl

# Configuração da ZED 2i
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30  # FPS da câmera

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"Erro ao abrir a ZED: {status}")
    exit(1)

runtime_parameters = sl.RuntimeParameters()
mat = sl.Mat()

# Caminho para o modelo YOLOv5 treinado
model_path = 'C:/Users/Lucas/Teste ZP/treinamento/yolov5/runs/train/yolov5x_results/weights/best.pt'

# Verificar se uma GPU está disponível e configurar o dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar o modelo YOLOv5 treinado
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.to(device)  # Mover o modelo para a GPU se disponível

# Dicionário de classes
class_labels = {
    0: 'DynamicsLifeForm',
    1: 'DynamicsObjects',
    2: 'StaticLifeForm',
    3: 'StaticObject'
}

class_colors = {
    'DynamicsLifeForm': (0, 255, 0),  # Verde
    'DynamicsObjects': (0, 0, 255),   # Vermelho
    'StaticLifeForm': (255, 0, 0),    # Azul
    'StaticObject': (255, 255, 0)     # Amarelo
}

zone_radius = {
    'DynamicsLifeForm': 60,  # Zona maior
    'DynamicsObjects': 30,   # Zona menor
    'StaticLifeForm': 0,     # Sem zona específica
    'StaticObject': 0        # Sem zona específica
}

def draw_boxes_and_zones(image, results, depth_image):
    detections = results.xyxy[0].cpu().numpy()  # Resultados da detecção

    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(cls)
        confidence = conf
        label = class_labels.get(class_id, 'Unknown')
        color = class_colors.get(label, (255, 255, 255))  # Branco como cor padrão
        radius = zone_radius.get(label, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f'{label}: {confidence:.2f}'
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if radius > 0:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            distance = depth_image.get_value(center_x, center_y)[1] / 1000  # Converter para metros

            cv2.circle(image, (center_x, center_y), int(radius / distance), color, 1)

    return image

def process_zed():
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            zed.retrieve_measure(mat, sl.MEASURE.DEPTH)

            image = mat.get_data()
            depth_image = mat.get_data()

            img_tensor = torch.from_numpy(image).float().to(device) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            results = model(img_tensor)

            image_with_boxes_and_zones = draw_boxes_and_zones(image, results, depth_image)

            cv2.imshow('YOLOv5 + ZED 2i Detection', image_with_boxes_and_zones)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_zed()
```

### 4. **Arquivo `requirements.txt`**

Crie um arquivo `requirements.txt` com as dependências necessárias:

```
torch
opencv-python-headless
ultralytics
```


## Configuração do Ambiente

### 1. Clone o Repositório YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
```

### 2. Instale as Dependências

Crie um ambiente virtual Python e instale as dependências:

```bash
python -m venv yolov5-env
source yolov5-env/bin/activate  # Linux
yolov5-env\Scripts\activate  # Windows

pip install -r requirements.txt
pip



 install opencv-python-headless
```

## Classificação e Armazenamento de Imagens

### Execute o Script de Classificação

```bash
python scripts/classify_images.py
```

## Detecção em Tempo Real com a Webcam

### Execute o Script de Detecção

```bash
python scripts/detect_webcam.py
```


Pressione `q` para sair da janela de detecção em tempo real.
```

### 6. **Executando o Projeto**

#### 6.1. **Classificação de Imagens**

Execute o script de classificação:

```bash
python scripts/classify_images.py
```

#### 6.2. **Detecção em Tempo Real com a Webcam**

Execute o script de detecção:

```bash
python scripts/detect_webcam.py
```

Agora você tem um projeto completo de detecção de zonas proxêmicas usando YOLOv5, com classificação de imagens e detecção em tempo real com a webcam!

## Contribuindo

Se você deseja contribuir com este projeto, por favor siga os passos abaixo:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nome-da-feature`)
3. Faça commit das suas alterações (`git commit -m 'Adiciona nova feature'`)
4. Faça push para a branch (`git push origin feature/nome-da-feature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.


