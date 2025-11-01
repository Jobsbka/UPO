import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GeometricTensor:
    """Мультивектор в Cl(3,0) - 8 компонент"""
    def __init__(self, components):
        # components: [скаляр, e1, e2, e3, e12, e13, e23, e123]
        self.components = components
    
    def geometric_product(self, other):
        a = self.components
        b = other.components
        
        # Таблица умножения для Cl(3,0)
        c0 = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7]
        c1 = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] - a[3]*b[5] + a[4]*b[2] + a[5]*b[3] - a[6]*b[7] - a[7]*b[6]
        c2 = a[0]*b[2] + a[1]*b[4] + a[2]*b[0] - a[3]*b[6] - a[4]*b[1] + a[5]*b[7] + a[6]*b[3] + a[7]*b[5]
        c3 = a[0]*b[3] + a[1]*b[5] + a[2]*b[6] + a[3]*b[0] - a[4]*b[7] - a[5]*b[1] - a[6]*b[2] - a[7]*b[4]
        c4 = a[0]*b[4] + a[1]*b[2] - a[2]*b[1] + a[3]*b[7] + a[4]*b[0] - a[5]*b[6] + a[6]*b[5] + a[7]*b[3]
        c5 = a[0]*b[5] + a[1]*b[3] - a[2]*b[7] - a[3]*b[1] + a[4]*b[6] + a[5]*b[0] - a[6]*b[4] - a[7]*b[2]
        c6 = a[0]*b[6] + a[1]*b[7] + a[2]*b[3] - a[3]*b[2] - a[4]*b[5] + a[5]*b[4] + a[6]*b[0] + a[7]*b[1]
        c7 = a[0]*b[7] + a[1]*b[6] - a[2]*b[5] + a[3]*b[4] + a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0]
        
        return GeometricTensor([c0, c1, c2, c3, c4, c5, c6, c7])

class GeometricLinear(nn.Module):
    """Геометрический линейный слой"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Веса как мультивекторы
        self.weight = nn.Parameter(torch.randn(out_features, in_features, 8))
        self.bias = nn.Parameter(torch.randn(out_features, 8))
        
    def forward(self, x):
        # x: [batch, in_features, 8]
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, 8)
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                # Геометрическое произведение входов с весами
                x_geom = GeometricTensor([x[:, j, k] for k in range(8)])
                w_geom = GeometricTensor([self.weight[i, j, k] for k in range(8)])
                product = x_geom.geometric_product(w_geom)
                
                for k in range(8):
                    output[:, i, k] += product.components[k]
            
            # Добавляем bias
            for k in range(8):
                output[:, i, k] += self.bias[i, k]
                
        return output

class GeometricReLU(nn.Module):
    """Геометрическая функция активации"""
    def forward(self, x):
        # x: [batch, features, 8]
        # Применяем ReLU к каждой компоненте отдельно
        return F.relu(x)

class SimpleGeometricNet(nn.Module):
    """Простая геометрическая нейросеть для классификации 3D объектов"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.geom_linear1 = GeometricLinear(3, 16)  # 3D точки -> 16 геометрических признаков
        self.relu1 = GeometricReLU()
        self.geom_linear2 = GeometricLinear(16, 32)
        self.relu2 = GeometricReLU()
        self.geom_linear3 = GeometricLinear(32, num_classes)
        
    def forward(self, x):
        # x: [batch, 3, 8] - 3D точки как мультивекторы
        x = self.geom_linear1(x)
        x = self.relu1(x)
        x = self.geom_linear2(x)
        x = self.relu2(x)
        x = self.geom_linear3(x)
        
        # Берем скалярную часть для классификации
        return x[:, :, 0]  # [batch, num_classes]

# Пример использования
def points_to_geometric(points):
    """Преобразование 3D точек в геометрические тензоры"""
    # points: [batch, num_points, 3]
    batch_size, num_points, _ = points.shape
    
    geometric_tensors = torch.zeros(batch_size, num_points, 8)
    
    # Скалярная часть = 1
    geometric_tensors[:, :, 0] = 1.0
    
    # Векторные части = координаты точек
    geometric_tensors[:, :, 1] = points[:, :, 0]  # x -> e1
    geometric_tensors[:, :, 2] = points[:, :, 1]  # y -> e2  
    geometric_tensors[:, :, 3] = points[:, :, 2]  # z -> e3
    
    return geometric_tensors

# Тестирование
if __name__ == "__main__":
    # Создаем сеть
    net = SimpleGeometricNet(num_classes=3)
    
    # Тестовые данные: 10 точек в 3D
    points = torch.randn(2, 10, 3)  # [batch, points, xyz]
    geometric_input = points_to_geometric(points)
    
    # Прямой проход
    output = net(geometric_input)
    print(f"Input shape: {points.shape}")
    print(f"Geometric input shape: {geometric_input.shape}") 
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")