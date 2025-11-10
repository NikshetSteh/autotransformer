# Auto Transformer
This is tool for fast prepare data:
- Apply scalers for nums features
- Extract embeddings for text 
- Encode labels columns


## Example
```python
transformer = MLPreprocessor(target_cols=["Target"])
df_x, df_y = transformer.fit_transform(df)
transformer.save("transformer.pkl")
```
