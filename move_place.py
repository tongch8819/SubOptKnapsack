import os

if __name__ == "__main__":
    src = ".\\result\\archive-7\\facebook\\1000"
    dst = ".\\result\\archive-5\\facebook\\1000"

    for dir in os.walk(src):
        sep = '\\'
        seed = dir[0].rsplit(sep, 1)[1]
        for file in os.listdir(dir[0]):
            src_path = os.path.join(dir[0], file)
            dst_path = os.path.join(dst, seed, file)

            if os.path.isfile(src_path) and not os.path.exists(dst_path):
                # print(f"src:{src_path}, dst:{dst_path}, f:{file}")
                os.rename(src_path, dst_path)