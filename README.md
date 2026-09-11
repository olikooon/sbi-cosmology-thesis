# Воспроизведение SOTA-статьи: Jeffrey, Alsing & Lanusse (2020)

**Simulation-Based Inference для космологической параметрической инференции — дипломная работа**

Этот репозиторий содержит точное воспроизведение методологии из статьи:

> Jeffrey, N., Alsing, J., Lanusse, F. (2020). *"Likelihood-free inference with neural compression of DES SV weak lensing map statistics"*, MNRAS 501, 954. [arXiv:2009.08459](https://arxiv.org/abs/2009.08459)

## Зачем это нужно

Это часть подготовки к дипломной работе на тему **Simulation-Based Inference (SBI) для космологической параметрической инференции на симуляциях CAMELS**.


## Что воспроизведено

Полный pipeline статьи, **на оригинальных данных и коде авторов**:

- 74 mock-симуляции weak lensing convergence maps (octant DES SV), сопоставленные с $n(z)$ реального DES SV
- Summary statistics: power spectrum + peak counts (1332 mock-реализации)
- Предобученный нейронный компрессор (Keras, `joint_augmented_regressor_190620.h5`)
- Ensemble из 4 neural density estimators через **pyDELFI** (Alsing et al. 2019): 2× Conditional Masked Autoregressive Flow + 2× Mixture Density Network
- Grid-оценка posterior (96×96 точек) + MCMC-сэмплирование (emcee) поверх обученного ensemble
- Применение к реальному наблюдению DES Science Verification

## Результат

| Параметр | Получено (это воспроизведение) | Published (Jeffrey et al. 2020) |
|---|---|---|
| Ω_m | 0.304 ± 0.165 | ≈ 0.3 |
| S₈ | 0.728 ± 0.091 | ≈ 0.75–0.8 |

Результаты согласуются с published значениями в пределах статистических ошибок.

## Технические детали: адаптация под современное окружение

Оригинальный код 2019–2020 года написан под TensorFlow 1.x и старые версии NumPy/SciPy/ChainConsumer. Для запуска на современном стеке (Kaggle, TensorFlow 2.20, NumPy 2.0, Python 3.12) потребовался ряд точечных патчей **без изменения научной логики**:

- `tf.placeholder`, `tf.Session`, `tf.train.AdamOptimizer` и др. → алиасы на `tf.compat.v1.*`
- `tf.contrib.distributions.fill_triangular` → `tensorflow_probability.math.fill_triangular` (модуль `tf.contrib` полностью удалён в TF2)
- `np.infty` → `np.inf` (удалено в NumPy 2.0)
- `scipy.integrate.simps` → `scipy.integrate.simpson` (переименовано)
- `chainconsumer` закреплён на версии `<1.0` (в 1.0+ полностью сменился API)
- Загрузка `.h5`-модели через `compile=False` (несовместимость формата сериализации Keras 3 со старым compile-конфигом)
- Разделение workflow на **eager-режим** (для Keras-компрессии) и **TF1 graph-режим** (обязателен для pyDELFI) — выполняются строго последовательно в рамках одной сессии, так как переключение необратимо

Полный список патчей и порядок их применения — в комментариях к ячейкам notebook'а.

## Структура репозитория

```
.
├── README.md                    — этот файл
├── sbi-cosmology-thesis.ipynb   — полный notebook с воспроизведением
```

Данные и код самой статьи (74 симуляции, компрессор, pyDELFI) клонируются автоматически из официального репозитория авторов: [github.com/NiallJeffrey/Likelihood-free_DES_SV](https://github.com/NiallJeffrey/Likelihood-free_DES_SV)

## Как запустить

Notebook рассчитан на среду Kaggle Notebooks (Python 3.12, доступ в интернет включён):

1. Импортировать `.ipynb` в новый Kaggle Notebook
2. Settings → Internet → On
3. Settings → Accelerator → **None / CPU** (GPU не требуется и создаёт конфликт eager/graph контекстов)
4. Выполнять ячейки последовательно (не Run All) — между разделами 4 и 5 есть необратимое переключение режима TensorFlow

## Связь с основной частью дипломной работы

Архитектурный паттерн, изученный и воспроизведённый здесь (компрессор → density estimator → coverage-валидация), применяется в основной части работы к:

- Систематическому сравнению методов SBI (NPE / NLE / NRE) на данных CAMELS
- Оценке робастности обученных моделей к misspecification гидродинамической физики (обучение на CAMELS-IllustrisTNG, тестирование на CAMELS-SIMBA)

с использованием современной библиотеки [`sbi`](https://github.com/sbi-dev/sbi) (Tejero-Cantero et al. 2020) вместо pyDELFI.

## Ссылки

- Оригинальная статья: [arXiv:2009.08459](https://arxiv.org/abs/2009.08459)
- Официальный репозиторий: [NiallJeffrey/Likelihood-free_DES_SV](https://github.com/NiallJeffrey/Likelihood-free_DES_SV)
- pyDELFI: [justinalsing/pydelfi](https://github.com/justinalsing/pydelfi)
- Alsing et al. (2019), "Fast likelihood-free cosmology with neural density estimators and active learning", MNRAS 488, 4440. [arXiv:1903.00007](https://arxiv.org/abs/1903.00007)
