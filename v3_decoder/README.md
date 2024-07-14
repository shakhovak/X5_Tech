# Baseline-проект

## Запуск

##### Склонируйте репозиторий

```
git clone https://git.codenrock.com/xxxxxxxx/xxxxx/xxxxxx.git x5_ner
cd x5_ner
```

##### Изменение решения

Внесите свои изменения в проект, модели, зависимости. Любые изменения направленные на улучшение результата работы
проекта.

Обратите внимание, система настроена таким образом, что тестовый датасет должен иметь название и располагаться строго
в `data/test.json`.
Результат работы (предсказание) тоже должно иметь строгое название и путь: `data/subsmission.json`.

##### Обучение модели

Обучение модели проходит на локальных устройствах участников. Для примера, вы можете использовать
скрипт `train_baseline.py` и самостоятельно разбить `train.json` на обучающие и тестовые выборки для локальных тестов

После работы, обучите модель на полном наборе `train.json` и отправьте решение на проверку.

##### Сборка решения
```shell
docker build . -t x5_ner
```

##### сделайте git push с весами

```shell
git add .
git commit -a -m"my solution"
git push
```

После команды `git push` в `main` ветку CI/CD система GitLab начнет автоматическую сборку проекта. По итогу, на странице
с задачей появится кнопка "**Проверить решение**". Нажмите её для запуска инференса.

Это займёт какое-то время. Обратите внимание, что все пакеты, модели и другие данные, которые, которые вы хотите
загрузить из интернета и использовать в проекте, должны быть загружены на этом этапе. В дальнейшем работа программы
будет оторвана от интернета и ни скачать, ни выкачать ничего не сможет.

Несмотря на то, что проверка решения выполняется на удаленном сервере, вы можете проверить запуск инференса и локально.
Если вы заранее поделили train датасет на условную обучающую и проверочную выборки для локально разработки, вы можете
положить соответствующие данные в папку `data/test.json` и запустить инференс локально.

##### Запуск инференса make submission py
В проверочной системе решение запускается следующим образом. Вы можете сымитировать это поведение локально.

```shell
docker run -it --network none --shm-size 2G --name x5_ner -v ./data:/app/data x5_ner python make_prediction.py
```

По итогу выполнения команды, файл с решением `submission.json` появится в папке `data`. Проверьте его на правильность
структуры. При желании, можете запустить расчет метрики на нём. Пример файла с расчетом метрики вы можете найти в корне
репозитория.