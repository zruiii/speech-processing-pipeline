import pandas as pd
import os
import subprocess
from typing import Optional, List, Tuple, Set
import glob
import time
import re

def sanitize_filename(filename: str) -> str:
    """
    按照规则净化文件名，移除或替换特殊字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        str: 处理后的文件名
    """
    # 创建替换映射
    replacements = {
        ':': '-',  # 英文冒号
        '：': '-', # 中文冒号
        '·': '-',  # 中文点号
        ',': '',   # 英文逗号
        '"': '',
        '.': '',
        '|': '-',
        '+': '-',
        '“': '',
        ' ': '',
        '”': '',
        '，': '',  # 中文逗号
        '、': '',  # 顿号
        '(': '',   # 英文左括号
        ')': '',   # 英文右括号
        '（': '',  # 中文左括号
        '）': '',  # 中文右括号
    }
    
    # 应用所有替换规则
    new_filename = filename
    for old_char, new_char in replacements.items():
        new_filename = new_filename.replace(old_char, new_char)
    
    return new_filename

def get_downloaded_episodes(book_dir: str) -> Set[int]:
    """
    获取目录中已下载的剧集编号
    
    Args:
        book_dir: 书籍目录路径
        
    Returns:
        Set[int]: 已下载的剧集编号集合
    """
    downloaded_episodes = set()
    if os.path.exists(os.path.join(book_dir, "NA.m4a")):
        return {1}
    
    # 检查所有音频文件
    audio_patterns = ["*.m4a", "*.mp3", "*.aac"]
    for pattern in audio_patterns:
        for file_path in glob.glob(os.path.join(book_dir, pattern)):
            try:
                # 从文件名提取剧集编号
                episode_num = int(os.path.splitext(os.path.basename(file_path))[0])
                downloaded_episodes.add(episode_num)
            except ValueError:
                continue
    
    return downloaded_episodes

def get_episode_info(url: str) -> Tuple[Optional[int], Optional[List[str]]]:
    """
    获取视频集合中的剧集数量和ID列表
    """
    try:
        print(f"正在获取视频信息，URL: {url}")
        
        command = ["yt-dlp", "--flat-playlist", "--print", "%(id)s", url]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            # 获取原始ID列表
            raw_ids = [id.strip() for id in result.stdout.strip().split('\n') if id.strip()]
            
            if not raw_ids:
                print("未获取到任何视频ID")
                return None, None
            
            # 为每个ID添加BV前缀（如果还没有）
            valid_ids = []
            for vid in raw_ids:
                if not vid.startswith('BV'):
                    vid = f'BV{vid}'
                valid_ids.append(vid)
            
            if valid_ids:
                return len(valid_ids), valid_ids
            else:
                print("警告：未找到有效的B站视频ID")
    except Exception as e:
        print(f"获取视频信息时发生错误: {str(e)}")
    return None, None

def download_bilibili_audio(url: str, book_name: str, format: str = "ba", output_dir: Optional[str] = None,
                          target_episodes: Optional[Set[int]] = None) -> bool:
    """
    下载B站视频的音频到指定文件夹，支持选择性下载特定剧集
    
    Args:
        url: B站视频链接
        book_name: 书名，用作存储文件夹名
        format: 音频格式，默认"ba"(best audio)
        output_dir: 输出的根目录，默认为当前目录
        target_episodes: 需要下载的剧集编号集合，None表示下载所有剧集
    
    Returns:
        bool: 下载是否成功
    """
    try:
        # 创建输出目录
        if output_dir is None:
            output_dir = os.getcwd()
        
        book_dir = os.path.join(output_dir, sanitize_filename(book_name))
        os.makedirs(book_dir, exist_ok=True)
        
        # 获取已下载的剧集
        downloaded_episodes = get_downloaded_episodes(book_dir)
        
        # 获取总集数和视频ID列表
        total_count, video_ids = get_episode_info(url)
        
        if total_count is None or not video_ids:
            print(f"无法获取 {book_name} 的视频信息")
            return False
            
        # 确定需要下载的剧集
        if target_episodes is None:
            # 下载所有未下载的剧集
            episodes_to_download = set(range(1, total_count + 1)) - downloaded_episodes
        else:
            # 只下载指定的未下载剧集
            episodes_to_download = target_episodes - downloaded_episodes
        
        if not episodes_to_download:
            print(f"{book_name} 已经下载完成，无需继续下载")
            return True
            
        print(f"开始下载 {book_name} 的以下剧集: {sorted(episodes_to_download)}")
        
        # 构建输出文件名模板
        output_template = os.path.join(book_dir, "%(playlist_index)s.%(ext)s")
        
        success_count = 0
        # 为每个需要下载的剧集构建下载命令
        for episode in sorted(episodes_to_download):
            if episode > len(video_ids):
                print(f"警告: 剧集 {episode} 超出范围，跳过")
                continue
                
            video_id = video_ids[episode - 1]
            episode_url = f"https://www.bilibili.com/video/{video_id}"
            
            # 直接指定输出文件名
            output_file = os.path.join(book_dir, f"{episode}.%(ext)s")
            
            command = [
                "yt-dlp",
                "-f", format,
                "-o", output_file,  # 使用新的输出文件名格式
                "--no-playlist",
                "--retries", "10",
                "--sleep-interval", "5",
                "--max-sleep-interval", "20",
                "--fragment-retries", "10",
                "--socket-timeout", "30",
                "--no-overwrites",
                "--continue",
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                episode_url
            ]
            
            print(f"\n正在下载第 {episode} 集...")
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            print(output.strip())
                    
                    return_code = process.poll()
                    if return_code == 0:
                        success_count += 1
                        break
                    else:
                        error = process.stderr.read()
                        print(f"下载第 {episode} 集失败: {error}")
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"等待 10 秒后进行第 {retry_count + 1} 次重试...")
                            time.sleep(10)
                        
                except Exception as e:
                    print(f"下载第 {episode} 集时发生错误: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"等待 10 秒后进行第 {retry_count + 1} 次重试...")
                        time.sleep(10)
            
            if retry_count == max_retries:
                print(f"第 {episode} 集在重试 {max_retries} 次后仍然失败")
            
            time.sleep(2)  # 下载间隔
        
        # 验证下载完成情况
        final_downloaded = get_downloaded_episodes(book_dir)
        if len(final_downloaded) >= total_count:
            print(f"{book_name} 下载完成！")
            return True
        else:
            remaining = total_count - len(final_downloaded)
            print(f"{book_name} 下载部分完成，成功下载 {success_count} 集，还有 {remaining} 集未完成")
            return False
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    with open('downloaded_books.txt', 'r', encoding='utf-8') as f:
        downloaded = list(f)

    downloaded = [x.strip() for x in downloaded]
    df = pd.read_excel("有声书2.xlsx")
    for index, row in df.iterrows():
        book_name = row['Book'].replace("《", '').replace("》", '')
        url = row['Link']

        if sanitize_filename(book_name) in downloaded:
            print(f"{book_name} 已下载，跳过")
            continue

        if "bilibili" in url:
            success = download_bilibili_audio(url, book_name, output_dir="audiobook")
        
            if success:
                print(f"{book_name} 下载成功完成！")
            else:
                print(f"{book_name} 下载过程中出现错误，请检查日志。")
            
            time.sleep(2)