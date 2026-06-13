PyTorch 学习笔记
===============

1. 为什么 criterion 作为 MSELoss 的实例对象可以像函数一样调用？

代码示例：
```python
x_clean = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ],
    dtype=torch.float32,
)
y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss() # 
criterion(model(x_clean), y)
```

在 Python 中，所有可以调用的对象（其中包括函数，对，函数也是对象）都遵循了相同的协议，即实现 `__call__` 方法。https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__call__

nn.MSELoss 继承 _Loss， 而 _Loss 继承 Module，Module 实现了 `__call__` 方法，通过 `_wrapped_call_impl` 最终调用 `forward` 方法。

2. pytorch 优化器的基本用法

```python
for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
```

3. CrossEntropyLoss + output_dim 为 1 会发生什么？

