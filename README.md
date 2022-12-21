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

Для сферы был написан скрипт на языке программирования C#:
```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
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

Затем у сферы были настроенны компоненты ```Decision Requester, Behavior Parameters``` и добавлен файл конфигурации нейронной сети:
```yaml
behaviors:
  RollerBall:
    trainer_type: ppo
    hyperparameters:
      batch_size: 10
      buffer_size: 100
      learning_rate: 3.0e-4
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
```

после чего был запущен ml agent из anacond prompt, ниже предоставленно gif с результатом работы проекта(нажмите для проигрывания):
![MLAgent (1)](https://user-images.githubusercontent.com/104893843/208933407-96fe41e8-5a23-4f14-97ca-2198cdd08c15.gif)

Далее тренировка была запущенна сразу на 9 копий, ниже предоставленно gif:
![9 копий (1)](https://user-images.githubusercontent.com/104893843/208936937-5d389992-2e47-41cf-967a-5adb8dc9b6f8.gif)

И соответственно на 27 копий, ниже предоставленно gif:
![27 копий (1) (1)](https://user-images.githubusercontent.com/104893843/208982271-9ea03535-9ace-43ed-9f65-db1b198d1b67.gif)


После ```60.000 step``` сферы стали достаточно плавно и точно перемещаться к цели по кратчайшему маршруту

## Задание 2
### Сделать описание конфигурационного
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

Был изменен прошлый и код стал следующим

```css
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public Transform Target2;

    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
        Target2.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(Target2.localPosition);
        sensor.AddObservation(this.transform.localPosition);
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
        float distanceToTarget2 = Vector3.Distance(this.transform.localPosition, Target2.localPosition);
        float distanceBetween = Vector3.Distance(Target.localPosition, Target2.localPosition) / 2;


        if(distanceToTarget > distanceToTarget2 - 0.4 && distanceToTarget < distanceToTarget2 + 0.4 && distanceToTarget2 < distanceBetween + 0.4)
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

после ```110 000 steps ``` шар хорошо научился двигаться в центр между кубами, и в случае когда у него не получалось встать между кубами из-за того что кубы появились слишком близко он пытается слететь с арены

Ниже приведен пример работы, gif
![шар в центр (1)](https://user-images.githubusercontent.com/104893843/208985997-575c957f-539d-434f-928e-02d993e13212.gif)


## Выводы
В ходе лабораторной работы я научился подключать ml agenta, создавать системы машинного обучения и интегрировать ее с Unity.
