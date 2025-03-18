// Получаем элементы страницы
const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d');
const clearButton = document.getElementById('clear-button');
const predictButton = document.getElementById('predict-button');
const resultSpan = document.getElementById('result');
const statusSpan = document.getElementById('status');
console.log('Входной тензор:', imageTensor.arraySync());
let isDrawing = false;

// Очищаем canvas белым цветом
function clearCanvas() {
  ctx.fillStyle = '#FFFFFF'; // Белый фон
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Обработчики событий для рисования
canvas.addEventListener('mousedown', () => isDrawing = true);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mousemove', draw);

function draw(event) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  ctx.fillStyle = '#000000'; // Чёрный цвет
  ctx.fillRect(x, y, 15, 15); // Рисуем квадраты размером 15x15
}

// Загружаем модель как Graph Model
let model;
(async function loadModel() {
  try {
    const modelUrl = './web_model/model.json'; // Убедитесь, что путь правильный
    model = await tf.loadGraphModel(modelUrl);
    statusSpan.textContent = 'Модель загружена! Можно рисовать.';
  } catch (error) {
    console.error('Ошибка загрузки модели:', error);
    statusSpan.textContent = 'Ошибка загрузки модели. Проверьте консоль.';
  }
})();

// Предобработка изображения
function preprocessCanvas(canvas) {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return tf.tidy(() => {
    let tensor = tf.browser.fromPixels(imageData, 1) // Градации серого (1 канал)
      .resizeNearestNeighbor([28, 28])               // Уменьшаем до 28x28 пикселей
      .toFloat()
      .div(255.0)                                    // Нормализуем значения пикселей
      .expandDims(0);                                // Добавляем размерность batch (форма [1, 28, 28, 1])
    
    console.log('Форма тензора:', tensor.shape);     // Отладочное сообщение
    return tensor;
  });
}

// Кнопка "Распознать"
predictButton.addEventListener('click', async () => {
  if (!model) {
    alert('Модель ещё не загружена. Пожалуйста, подождите.');
    return;
  }

  // Получаем предсказание
  const imageTensor = preprocessCanvas(canvas);
  console.log('Форма тензора:', imageTensor.shape);
  const prediction = await model.predict(imageTensor).data();
  const result = prediction.indexOf(Math.max(...prediction)); // Находим наиболее вероятную цифру
  resultSpan.textContent = result; // Показываем результат
});

// Кнопка "Очистить"
clearButton.addEventListener('click', () => {
  clearCanvas();
  resultSpan.textContent = '?';
});
