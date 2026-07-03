with open("diagnostic_visualization.ipynb", "rb") as f:
    chunk = f.read(500)
    print("Raw bytes chunk:")
    print(chunk)
    try:
        print("Decoded as UTF-8:")
        print(chunk.decode("utf-8"))
    except Exception as e:
        print("UTF-8 decode error:", e)
    try:
        print("Decoded as UTF-16:")
        print(chunk.decode("utf-16"))
    except Exception as e:
        print("UTF-16 decode error:", e)
