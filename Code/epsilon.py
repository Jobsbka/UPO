# epsilon_derivation.py
import numpy as np
from scipy import constants as const

def calculate_epsilon():
    """
    Расчёт параметра нелинейности ε ~ l_Planck / L_Universe
    """
    l_planck = np.sqrt(const.hbar * const.G / const.c**3)
    L_universe = const.c / (67.4 * 1000 / (const.parsec * 1e6))  # Хаббловский радиус
    
    epsilon = l_planck / L_universe
    return epsilon

def epsilon_effects():
    """Демонстрация эффектов ε в разных масштабах"""
    epsilon = calculate_epsilon()
    
    print(f"Параметр нелинейности ε: {epsilon:.2e}")
    print(f"Вклад в космологическую постоянную: ~{epsilon:.2e}")
    print(f"Поправки к квантовой механике: ~{epsilon**2:.2e}")
    
    # Время декогеренции для разных масс
    masses_kg = np.array([const.m_e, 1e-23, 1e-15, 1e-9, 1e-3])  # от электрона до пылинки
    masses_names = ["электрон", "молекула", "бактерия", "пылинка", "капля"]
    
    print("\nВремя декогеренции (в относительных единицах):")
    for mass, name in zip(masses_kg, masses_names):
        t_decoherence = (const.m_e / mass)**2
        print(f"{name:>10}: {t_decoherence:.2e}")
    
    return epsilon

# Добавляем запускающий код
if __name__ == "__main__":
    print("=== Расчёт параметра нелинейности ε ===")
    epsilon = epsilon_effects()
    print(f"\nФизические константы:")
    print(f"l_Planck = {np.sqrt(const.hbar * const.G / const.c**3):.2e} м")
    print(f"L_Universe = {const.c / (67.4 * 1000 / (const.parsec * 1e6)):.2e} м")