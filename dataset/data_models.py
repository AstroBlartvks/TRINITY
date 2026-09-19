from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Scaler:
    """Все преобразования в одном месте. Статистики посчитаны только по train."""
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float
    g_mean: float
    g_std: float

    def transform_x(self, log_x: np.ndarray) -> np.ndarray:
        return (log_x - self.x_mean) / self.x_std

    def transform_y(self, log_y: np.ndarray) -> np.ndarray:
        return (log_y - self.y_mean) / self.y_std

    def transform_g(self, log_g: np.ndarray) -> np.ndarray:
        return (log_g - self.g_mean) / self.g_std

    def inverse_y(self, z) -> np.ndarray:
        """z-score предсказания -> волатильность в исходных единицах."""
        z = np.asarray(z, dtype=np.float64)
        return np.exp(z * self.y_std + self.y_mean)


@dataclass
class Split:
    """Один кусок выборки (train / val / test)."""
    X: np.ndarray             # (N, window)  нормализованный log RV
    y: np.ndarray             # (N,)         нормализованный log RV_forward
    g: np.ndarray             # (N,)         нормализованный log GARCH-прогноз
    emb: np.ndarray           # (N, dim)     новостной эмбеддинг дня t
    y_raw: np.ndarray         # (N,)         сырая волатильность (для метрик)
    g_raw: np.ndarray         # (N,)         сырой прогноз GARCH (для бейзлайна)
    dates: pd.DatetimeIndex   # (N,)         день t, в который делается прогноз

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "Split(пусто)"
        return (f"Split(n={len(self)}, "
                f"{self.dates[0].date()} .. {self.dates[-1].date()})")

@dataclass
class VolDataset:
    train: Split
    val: Split
    test: Split
    scaler: Scaler
    meta: dict

    # сводка
    def summary(self) -> None:
        h = self.meta['horizon']
        print("=" * 62)
        print(f"Датасет: окно={self.meta['window']}д, горизонт={h}д, "
              f"эмбеддинг={self.meta['emb_dim']}")
        print(f"Таргет: RV за дни t+1..t+{h}, признаки — по день t включительно")
        print("-" * 62)
        for nm, sp in (('train', self.train), ('val', self.val), ('test', self.test)):
            if len(sp) == 0:
                print(f"  {nm:5s}: пусто")
                continue
            print(f"  {nm:5s}: n={len(sp):5d}  "
                  f"{sp.dates[0].date()} .. {sp.dates[-1].date()}  "
                  f"vol: {sp.y_raw.min():.5f} .. {sp.y_raw.max():.5f}")
        gf = self.meta['garch_fallbacks']
        if gf:
            print(f"  GARCH: фолбэк сработал {gf} раз "
                  f"({gf / max(1, self.meta['garch_days']):.1%} дней)")
        print("=" * 62)

    # torch-лоадеры
    def loaders(self, batch_size: int = 64, shuffle_train: bool = True):
        """Возвращает (train_loader, val_loader, test_loader).

        drop_last=True на train — иначе BatchNorm падает на батче из 1 элемента.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        def make(sp: Split, shuffle: bool, drop_last: bool):
            dsets = TensorDataset(
                torch.from_numpy(sp.X),
                torch.from_numpy(sp.y),
                torch.from_numpy(sp.g),
                torch.from_numpy(sp.emb),
            )
            return DataLoader(dsets, batch_size=batch_size,
                              shuffle=shuffle, drop_last=drop_last)

        return (make(self.train, shuffle_train, True),
                make(self.val, False, False),
                make(self.test, False, False))