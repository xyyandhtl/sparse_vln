def print_title(title: str, total_length: int = 120, separator_char: str = '-'):
    total_length = max(total_length, len(title) + 2)
    # 计算分隔符的左右长度
    side_length = (total_length - len(title)) // 2
    left = separator_char * side_length
    right = separator_char * (total_length - len(left) - len(title))
    # 打印分隔行
    print(f"{left}{title}{right}")