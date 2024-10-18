from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Carregar o modelo treinado com suas classes (você pode substituir pelo seu modelo)
model = YOLO('yolov8n.pt')  

# Lista de labels para as 4 doenças (você pode adaptar os nomes dos arquivos)
disease_labels = {
    'alternariaCucumerina': 'labels/alternariaCucumerina.txt',
    'antracnose': 'labels/antracnose.txt',
    'manchaAngular': 'labels/manchaAngular.txt',
    'mildew': 'labels/mildew.txt'
}

# Rota para a página inicial
@app.route('/')
def index():
    return render_template('index.html')

# Rota para o upload da imagem
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Fazer a predição na imagem usando YOLOv8
        results = model(filepath)
        
        # Carregar a imagem com OpenCV
        img = cv2.imread(filepath)

        # Variáveis para armazenar a doença mais provável e maior confiança
        best_disease = None
        highest_confidence = 0

        # Percorrer as 4 doenças e comparar com a imagem
        for disease, label_file_path in disease_labels.items():
            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as f:
                    annotations = f.readlines()

                # Iterar sobre as anotações para desenhar os bounding boxes e calcular similaridade
                for annotation in annotations:
                    class_id, x_center, y_center, width, height = map(float, annotation.split())
                    x1 = int((x_center - width / 2) * img.shape[1])
                    y1 = int((y_center - height / 2) * img.shape[0])
                    x2 = int((x_center + width / 2) * img.shape[1])
                    y2 = int((y_center + height / 2) * img.shape[0])
                    
                    # Aqui, você pode implementar uma lógica de comparação entre a imagem e as anotações

                    # Exemplo: adicionar confiança (aqui como exemplo, use a predição real)
                    confidence = 0.95  # Valor fictício, ajustar conforme sua lógica
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_disease = disease

                    # Desenhar o bounding box na imagem
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{disease} {confidence:.2f}', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Salvar a imagem com os bounding boxes
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{file.filename}")
        cv2.imwrite(output_path, img)
        print(f"Imagem salva em: {output_path}")  # Verifique no console se a imagem foi salva

        # Retornar o resultado para o usuário
        return render_template('index.html', filename=f"output_{file.filename}", disease=best_disease, confidence=highest_confidence)

# Rota para exibir a imagem processada
@app.route('/static/uploads/<filename>')
def display_image(filename):
    return url_for('static', filename=f'uploads/{filename}')

if __name__ == "__main__":
    app.run(debug=True)
