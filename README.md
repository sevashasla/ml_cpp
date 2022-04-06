# ML-CPP
Небольшая библиотека для машинного обучения, написанная на C++

## Установка
1. Скачайте репозиторий проекта:
```sh
    git clone https://github.com/sevashasla/ml_cpp.git
```
2. Затем выполните в репозитории:
```sh
mkdir build
cd build
cmake ..
make
```

## Примеры
Базовый пример использование (после подключения библиотеки через `cmake`):

 1. Регрессия:
```cpp
// создаём слои, в дальнейшем через них будут течь градиенты
nn::models::SingleLayer<double, nn::layers::Linear<double, 1, 1>> model;
nn::models::SingleLayer<double, nn::losses::MSELoss<double>> loss_fn;

// данные
Tensor<double> x(
    Matrix<double>({
        {1.0}, {2.0},
        {3.0}, {4.0},
        {5.0}
    }), nullptr, false);
Tensor<double> y(Matrix<double>({
        {2.0}, {4.0},
        {6.0}, {8.0},
        {10.0}}), 
        nullptr/*от кого возникло*/, 
        false/*нужно ли через него делать step*/);

// tqdm цикл, будет показывать progress_bar
tqdm_like::tqdm_like(
    0 /*начально*/, 
    10000 /*конец*/, 
    1 /*шаг*/,
    /*задача на исполнение в цикле for*/ [&](){

    // прямой проход
    auto pred = model.forward(x);

    // считаем ошибку
    auto loss = loss_fn.forward(y, pred);

    // обратный проход
    loss.backward();

    // делаем шаг, обновляем веса
    loss.make_step(1e-3);

    // зануляем градиенты
    loss.zero_grad();

    // ломаем граф, который строился при прямом проходе
    loss.break_graph();
});

auto pred = model.forward(x);
auto loss = loss_fn.forward(y, pred);

cout << "loss: " << loss;
}
```
2. Классификация
```cpp
// данные
Tensor<double> x(
		Matrix<double>({
		{1.5792128155073915},
		{0.6476885381006925},
		{-0.4694743859349521},
		{0.7674347291529088},
		{0.5425600435859647},
		{-0.23413695694918055},
		{-0.13826430117118466},
		{1.5230298564080254},
		{0.4967141530112327},
		{-0.23415337472333597}	
	}), nullptr, false);
// первоначальный y
Tensor<size_t> y_fresh(
    Matrix<size_t>({
        {1},
        {1},
        {0},
        {1},
        {1},
        {0},
        {0},
        {1},
        {1},
        {0}	
}), nullptr, false);
// преобразуем с помощью one-hot
Tensor<double> y(
    static_cast<Matrix<double>>(preprocessing::OneHot<2>(y_fresh)), 
    nullptr, 
    false);

// модель вида sequential - выход одного слоя 
// передаётся последовательно в другой.
auto model = nn::models::Sequential<
    double, // тип, с которым работаем
    nn::layers::Linear<double, 1 /*вход*/, 3/*выход*/>,
    nn::layers::BatchNorm<double>,
    nn::layers::ReLU<double>,
    nn::layers::Linear<double, 3/*вход*/, 2/*выход*/>
>();

// в качестве функии ошибки - CrossEntropyLoss
nn::models::SingleLayer<double, nn::losses::CrossEntropyLoss<double, 2/*число классов*/>> loss_fn;


tqdm_like::tqdm_like(0, 10000, 1, [&](){
    auto pred = model.forward(x);
    auto loss = loss_fn.forward(y, pred);
    loss.backward();
    loss.make_step(1e-3);
    loss.zero_grad();
    loss.break_graph();
});

auto pred = model.forward(x);
auto loss = loss_fn.forward(y, pred);

// считает, какой класс наиболее вероятный
auto pred_classes = postprocessing::predict<2>(pred);

cout << "loss: " << loss;

// точность
cout << "accuracy: " << metrics::accuracy<double>(static_cast<Matrix<size_t>&>(y_fresh), pred_classes);

// вероятности каждого класса
cout << "probabilities:\n";
cout << postprocessing::predictProba<2, double>(pred);
```

В тестах можно увидеть и другие примеры использования.
# Проект
Есть базовый шаблонный класс *Matrix*, от него отнаследован шаблонный класс *Tensor*, который имеет возможность автоматического дифференцирования. При простых арифметических операциях появляется новый потомок класса *Layer*, который имеет информацию о *Tensor*(ах), из которых он появился. После выполнения операции из *Layer* появляется новый *Tensor*, который имеет информацию о том, откуда он появился. Таким образом при вызове `.backward()` течёт градиент - от тензора к слою, а от слоя дальше к тензорам.

В комментариях к классам я постарался пояснить откуда появились те или иные формулы.

# Классы и структуры
В большинстве классов и функций один из шаблонных параметров - в каких числах считать результат.

Базовые типы:
- Шаблонный класс Matrix. Шаблонные параметры
    - Числа, в которых считать
- Шаблонный класс Tensor, наследован от Matrix

Функции активации:
- Шаблонный класс ReLU
- Шаблонный класс LeakyReLU
- Шаблонный класс Sigmoid

Слои:
- Шаблонный класс BatchNorm
- Шаблонный класс Linear - линейный слой. Шаблонные параметры:
    - Числа, в которых считать
    - Вход линейного слоя
    - Выход линейного слоя

Функции ошибок:
- Шаблонный класс MSELoss.
- Шаблонный класс CrossEntropyLoss

Модели:
- Шаблонный класс SigleLayer. Шаблонные параметры:
    - Слой, от которого создаётся
- Шаблонный класс Sequential - последовательное выполнение прямого пути - выход одного слоя подаётся во вход другого. Шаблонные параметры:
    - Слои, от которых создаётся

Метрики:
- Шаблонная функция accuracy
- Шаблонная функция meanAveragePercentageError

Postprocessing:
- Шаблонная функция predictProba - считает вероятность класса через их логиты через SoftMax. Шаблонные параметры:
    - Количество классов
    - Числа, в которых считать
- Шаблонная функция predict - считает классы через их логиты. Шаблонные параметры:
    - Количество классов
    - Числа, в которых считать

Preprocessing:
- Шаблонная функция OneHot. Шаблонные параметры:
    - Количество классов
    - Числа, в которых считать

tqdm_like:
- Функция tqdm_like - показывает progress bar и приблизительное время, которое осталось до завершения. Первые три аргументы - (начало, конец, шаг) как в цикле for, последний - функция, которую нужно выполнять в цикле каждый раз.


## TODO:
- add BatchCount
- requires_grad field
- add optimizer
- add MAE
- add initialization
- softmax Layer
- add gtest
