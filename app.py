from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = YOLO('yolov8n.pt')  # Você precisará treinar seu próprio modelo YOLOv8 para doenças em folhas de melancia

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
        
        # Iterar sobre os resultados para desenhar os bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas do bounding box
                confidence = box.conf[0]  # Confiança
                class_name = model.names[int(box.cls[0])]  # Nome da classe (doença)

                # Desenhar o bounding box e a confiança na imagem
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{class_name} {confidence:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Salvar a imagem com os bounding boxes
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{file.filename}")
        cv2.imwrite(output_path, img)

        return render_template('index.html', filename=f"output_{file.filename}")

# Rota para exibir a imagem processada
@app.route('/uploads/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == "__main__":
    app.run(debug=True)
