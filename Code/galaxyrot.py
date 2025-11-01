# galaxy_rotation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as const

def calculate_a0(H0=67.4):
    """Расчёт a₀ для использования в этом файле"""
    H0_SI = H0 * 1000 / (const.parsec * 1e6)
    return const.c * H0_SI / (2 * np.pi)

def newtonian_velocity(M, r):
    """Ньютоновская кривая вращения"""
    G = 6.67430e-11
    return np.sqrt(G * M / r)

def MOND_velocity(M, r, a0):
    """Модифицированная динамика в пределе MOND"""
    G = 6.67430e-11
    v_newt = np.sqrt(G * M / r)
    # Интерполяционная функция μ(x) = x/√(1 + x²)
    x = v_newt**2 / (r * a0)
    mu = x / np.sqrt(1 + x**2)
    return v_newt / np.sqrt(mu)

def simulate_galaxy_rotation():
    """Симуляция кривой вращения для типичной спиральной галактики"""
    M = 1e41  # кг (масса видимой материи ~5e10 M☉)
    r = np.logspace(18, 21, 100)  # м (радиус от 1 до 100 кпк)
    a0 = calculate_a0()
    
    v_newton = newtonian_velocity(M, r)
    v_mond = MOND_velocity(M, r, a0)
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.semilogx(r, v_newton, 'r--', label='Ньютоновская (только видимая масса)')
    plt.semilogx(r, v_mond, 'b-', label='ЕПО (MOND предел)')
    plt.xlabel('Радиус (м)')
    plt.ylabel('Скорость вращения (м/с)')
    plt.legend()
    plt.title('Кривые вращения галактик в ЕПО')
    plt.grid(True, alpha=0.3)
    
    # Добавим примерные масштабы
    r_kpc = r / (3.086e19)  # перевод в килопарсеки
    plt.twiny()
    plt.semilogx(r_kpc, v_mond, alpha=0)  # невидимая кривая для второй оси
    plt.xlabel('Радиус (кпк)')
    
    plt.tight_layout()
    plt.show()
    
    # Вывод численных результатов
    print(f"Масса галактики: {M:.2e} кг (~{M/2e30:.1f} × 10^10 M☉)")
    print(f"a₀ = {a0:.2e} м/с²")
    print(f"Скорость на 10 кпк: Ньютон = {newtonian_velocity(M, 10*3.086e19):.0f} м/с, ЕПО = {MOND_velocity(M, 10*3.086e19, a0):.0f} м/с")

# Добавляем запускающий код
if __name__ == "__main__":
    print("=== Симуляция кривых вращения галактик ===")
    simulate_galaxy_rotation()