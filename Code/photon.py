# photon_geometry.py
import numpy as np
import matplotlib.pyplot as plt

def photon_wavefunction(x, t, omega, k):
    """
    Геометрическая волновая функция фотона
    Ψ = стоячая_волна ⊗ бегущая_волна
    """
    # Стоячая волна (форма)
    standing_wave = np.exp(-x**2) * np.cos(omega * t)
    
    # Бегущая волна (распространение)
    traveling_wave = np.exp(1j * (k * x - omega * t))
    
    return standing_wave * traveling_wave

def visualize_photon_geometry():
    """3D визуализация геометрии фотона"""
    x = np.linspace(-5, 5, 100)
    t = np.linspace(0, 4*np.pi, 100)
    X, T = np.meshgrid(x, t)
    
    Z = photon_wavefunction(X, T, omega=1, k=1)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Амплитуда
    ax1 = fig.add_subplot(131)
    contour1 = ax1.contourf(X, T, np.abs(Z), levels=50, cmap='viridis')
    plt.colorbar(contour1, ax=ax1)
    ax1.set_title('Амплитуда фотона\n(стоячая волна)')
    ax1.set_xlabel('Пространство')
    ax1.set_ylabel('Время')
    
    # Фаза
    ax2 = fig.add_subplot(132)
    contour2 = ax2.contourf(X, T, np.angle(Z), levels=50, cmap='hsv')
    plt.colorbar(contour2, ax=ax2)
    ax2.set_title('Фаза фотона\n(бегущая волна)')
    ax2.set_xlabel('Пространство')
    ax2.set_ylabel('Время')
    
    # 3D визуализация
    ax3 = fig.add_subplot(133, projection='3d')
    surf = ax3.plot_surface(X, T, np.real(Z), cmap='coolwarm', alpha=0.8)
    ax3.set_title('Вещественная часть волновой функции')
    ax3.set_xlabel('Пространство')
    ax3.set_ylabel('Время')
    ax3.set_zlabel('Амплитуда')
    
    plt.tight_layout()
    plt.show()
    
    print("Геометрическая модель фотона в ЕПО:")
    print("- Стоячая волна задаёт форму 'пузыря'")
    print("- Бегущая волна определяет распространение")
    print("- Комбинация объясняет дуальность волна-частица")

# Добавляем запускающий код
if __name__ == "__main__":
    print("=== Визуализация геометрии фотона ===")
    visualize_photon_geometry()