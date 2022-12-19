# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
- Упаев Кирилл Анатоьлевич 
- РИ-210943
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.


## Цель работы
познакомиться с программными средствами для создания
системы машинного обучения и ее интеграции в Unity.

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity.
Ход работы:
Был создан пустой 3D проект на Unity, к ней был подключен ML Agent
Далее при помощи anaconda prompt били скачаны ```mlagents, torch```

В проекте были созданны сфера, плоскость и куб

Для сферы был написан код на ЯП C#: 
``css
использование системы.Коллекции;
использование системы.Коллекции.Общий;
использование UnityEngine;
используя Unity.Агенты;
используя Unity.Агенты.Датчики;
используя Unity.Агенты.Приводы;

RollerAgent публичного класса: агент
{
 Твердое тело rBody;
 // Start вызывается перед первым обновлением кадра
 аннулирование запуска()
    {
 rBody = GetComponent<Жесткое тело>();
    }

 цель общественного преобразования;
 публичное переопределение аннулирует OnEpisodeBegin()
    {
 если (это.преобразовать.localPosition.y < 0)
        {
 это.rBody.angularVelocity = Вектор3.ноль;
 это.Тело.скорость = Вектор3.ноль;
 this.transform.localPosition = новый Вектор3(0, 0.5f, 0);
        }

 Target.localPosition = новый Вектор3 (случайное значение * 8 - 4, 0,5f, случайное значение * 8 - 4);
    }
 публичное переопределение аннулирует наблюдения за сбором данных (VectorSensor sensor)
    {
 датчик.Дополнительное наблюдение (Target.localPosition);
 датчик.AddObservation(this.transform.localPosition) Дополнительное наблюдение(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```

Затем у сферы были настроенны компоненты ```Decision Requester, Behavior Parameters``` и добавлен файл конфигурации нейронной сети
``ямл
поведение:
 РоллерБолл:
 тип обучаемого: ppo
 гиперпараметры:
 размер пакета: 10
 размер буфера: 100
 learning_rate: 3,0e-4
 бета-версия: 5.0e-4
 эпсилон: 0,2
 лямбда: 0,99
 количество страниц: 3
 расписание обучения: линейное
 сетевые настройки:
 нормализовать: false
 скрытых единиц: 128
 num_layers: 2
 награды_сигналы:
 внешние:
 гамма: 0,99
 прочность: 1,0
 максимальные шаги: 500000
 временной горизонт: 64
 суммарная частота: 10000
```

после чего был запущен ml agent из anacond prompt, ниже предоставленно gif с результатом работы проекта(нажмите для проигрывания)
![2022-10-26-20-13-47](https://user-images.githubusercontent.com/49115035/198072629-7f8452af-baf0-4bae-a9d8-76042611bf73.gif)


Далее тренировка была запущенна сразу на 9 копий, ниже предоставленно gif
![2](https://user-images.githubusercontent.com/49115035/198074291-e3813dfb-ceea-4389-a264-9fda656308ba.gif)

И соответственно на 27 копий, ниже предоставленно gif
![3](https://user-images.githubusercontent.com/49115035/198075173-174d5d46-23d5-4512-a0a8-fd88e68ab0c8.gif)


После ```60.000 step``` сферы стали достаточно плавно и точно перемещаться к цели по кратчайшему маршруту

## Задание 2
### Сделать описание конфигурационного [файла](https://github.com/VenchasS/DA-in-GameDev-lab3/blob/main/rollerball_config.yaml)
```yaml
 behaviors: 
  RollerBall: #Behaviour Name из компоненты Behaviour Parameters
    trainer_type: ppo #тип тренировки ppo (Proximal Policy Optimization)
    hyperparameters: #гипер параметры
      batch_size: 10 #количество опытов на каждой итерации градиентного спуска
      buffer_size: 100 #количество опыта, который необходимо собрать перед обновлением модели
      learning_rate: 3.0e-4 #начальная скорость обучения для градиентного спуска
      beta: 5.0e-4 #cила регуляризации энтропии, которая позволяет делаеть политику "более случайной"
      epsilon: 0.2 #влияет на то, насколько быстро политика может развиваться во время обучения
      lambd: 0.99 #параметр регуляризации, используемый при расчете обобщенной оценки преимущества
      num_epoch: 3 #количество проходов через буфер опыта при выполнении оптимизации градиентного спуска
      learning_rate_schedule: linear #определяет, как скорость обучения будет менятся с течением времени.
    network_settings: #настройки
      normalize: false #Применяется ли нормализация к входным данным векторного наблюдения.
      hidden_units: 128 #Количество юнитов в скрытых слоях нейронной сети.
      num_layers: 2 #количество слоев
    reward_signals: #настройки наград
      extrinsic: #внешние награды
        gamma: 0.99 #Коэффициент дисконтирования для будущих вознаграждений, поступающих от окружающей среды.
        strength: 1.0 #Фактор, на который можно умножить вознаграждение, получаемое от окружающей среды
    max_steps: 500000 #максимальное количество итераций
    time_horizon: 64 #Сколько шагов опыта нужно собрать для каждого агента, прежде чем добавлять его в буфер опыта.
    summary_freq: 10000 #Количество опыта, которое необходимо собрать перед созданием и отображением статистики обучения.
```


## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар
перемещался между двумя кубами разного цвета. Кубы должны, как и в первом
задании, случайно изменять координаты на плоскости.

Был изменен прошлый [скрипт](https://github.com/VenchasS/DA-in-GameDev-lab3/blob/main/RollerAgent.cs) и код стал следующим
```css
использование системы.Коллекции;
использование системы.Коллекции.Общий;
использование UnityEngine;
используя Unity.Агенты;
используя Unity.Агенты.Датчики;
используя Unity.Агенты.Приводы;

RollerAgent публичного класса: агент
{
 Твердое тело rBody;
 // Start вызывается перед первым обновлением кадра
 аннулирование запуска()
    {
 rBody = GetComponent<Жесткое тело>();
    }

 цель общественного преобразования;
 цель публичного преобразования2;

 публичное переопределение аннулирует OnEpisodeBegin()
    {
 если (это.преобразовать.localPosition.y < 0)
        {
 это.rBody.angularVelocity = Вектор3.ноль;
 это.Тело.скорость = Вектор3.ноль;
 this.transform.localPosition = новый Вектор3(0, 0.5f, 0);
        }

 Target.localPosition = новый Вектор3 (случайное значение * 8 - 4, 0,5f, случайное значение * 8 - 4);
 Target2.localPosition = новый Вектор3 (случайное значение * 8 - 4, 0,5f, случайное значение * 8 - 4);
    }
 публичное переопределение аннулирует наблюдения за сбором данных (VectorSensor sensor)
    {
 датчик.Дополнительное наблюдение (Target.localPosition);
 датчик.Дополнительное наблюдение (Target2.localPosition);
 датчик.Дополнительное наблюдение (this.transform.localPosition);
 датчик.Дополнительное наблюдение (rBody.velocity.x);
 датчик.Дополнительное наблюдение (rBody.velocity.z);
    }
 открытый плавающий умножитель силы = 10;
 публичное переопределение аннулирует полученное действие (ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
 controlSignal.z = буферы действий.Непрерывные действия[1];
 Тело.Добавить силу(controlSignal * Умножитель силы);

 float distanceToTarget = Вектор3.Расстояние(this.transform.localPosition, Target.localPosition);
 float distanceToTarget2 = Вектор3.Расстояние(this.transform.localPosition, Target2.localPosition);
 float distanceBetween = Вектор3.Расстояние(Target.localPosition, Target2.localPosition) / 2;


 если(distanceToTarget > distanceToTarget2 - 0.4 && distanceToTarget <distanceToTarget2 + 0.4 && distanceToTarget2 < Расстояние между + 0.4)
        {
 Отклонение от заданного значения (1,0f);
 Конечный код();
        }
 иначе, если (this.transform.localPosition.y < 0)
        {
 Конечный код();
        }
    }
}
```

после ```110 000 steps ``` шар хорошо научился двигаться в центр между кубами, и в случае когда у него не получалось встать между кубами из-за того что кубы появились слишком близко он пытается слететь с арены


Ниже приведен пример работы, gif
![4](https://user-images.githubusercontent.com/49115035/198097247-84f5d66c-41d8-4320-9805-43043c2a9273.gif)


## Выводы
В ходе лабораторной работы я научился подключать ml agenta, создавать системы машинного обучения и интегрировать ее с Unity.