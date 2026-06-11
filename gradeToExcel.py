#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def normalize_venue_name(venue_name):
    """レース場名を正規化（琵琶湖→びわこ）"""
    if venue_name:
        venue_name = venue_name.replace('琵琶湖', 'びわこ')
    return venue_name

def try_convert_number(value):
    """文字列を数値に変換試行"""
    if value is None or value == '' or value == '.' or value == '.  .':
        return value  # そのまま返す
    
    value_str = str(value).strip()
    
    # ドット区切り（レースタイムの場合）
    if '.' in value_str and value_str.count('.') >= 2:
        return value_str  # レースタイム形式は文字列のまま
    
    # 整数変換試行
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # 浮動小数点数変換試行
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # 変換失敗時は文字列で返す
    return value_str

def parse_venue_section(lines, start_idx, end_idx):
    """1つの開催地セクション（KBGN～KEND）を解析"""
    section_lines = lines[start_idx:end_idx]
    
    # 基本情報の抽出
    race_venue = None
    race_date = None
    race_day = None
    race_title = None
    
    for line in section_lines[:50]:
        if 'ボートレース' in line:
            match = re.search(r'ボートレース(.+?)[\s　]*$', line)
            if match:
                race_venue = match.group(1).replace('　', '').strip()
                # レース場名を正規化
                race_venue = normalize_venue_name(race_venue)
        
        if '第' in line and '日' in line and '/' in line:
            match = re.search(r'第\s*(\d+)\s*日\s+(\d{4})/\s*(\d{1,2})/\s*(\d{1,2})', line)
            if match:
                race_day = int(match.group(1))
                race_date = f"{match.group(2)}/{int(match.group(3)):02d}/{int(match.group(4)):02d}"
        
        if '杯' in line or '戦' in line:
            match = re.search(r'([\w\s\w]*[杯戦])', line)
            if match:
                title = match.group(1).replace('　', '').strip()
                if title and len(title) > 1:
                    race_title = title
    
    # レース情報の辞書
    races = {}
    
    # 「着 艇 登番」の行を探して各レースを処理
    for idx, line in enumerate(section_lines):
        if '着 艇 登番' in line:
            # この行の直前を見て気象情報を探す
            race_header_idx = idx - 1
            
            # レース情報行を遡る
            while race_header_idx >= 0:
                prev_line = section_lines[race_header_idx]
                race_match = re.match(r'\s*(\d+|[0-9]+)R\s+', prev_line)
                
                if race_match:
                    race_num = int(re.search(r'(\d+)', race_match.group(1)).group(1))
                    
                    # 既に存在するなら新しく作る
                    if race_num in races and races[race_num]['racers']:
                        race_header_idx -= 1
                        continue
                    
                    races[race_num] = {
                        'number': race_num,
                        'type': None,
                        'distance': None,
                        'weather': None,
                        'wind_direction': None,
                        'wind_speed': None,
                        'wave_height': None,
                        'racers': [],
                        'payouts': []
                    }
                    
                    # 気象情報を抽出
                    header_line = prev_line
                    
                    # レースタイプ
                    race_type_match = re.search(r'\d+R\s+(.+?)\s+H\d+m', header_line)
                    if race_type_match:
                        races[race_num]['type'] = race_type_match.group(1).replace('　', '').strip()
                    
                    # 距離
                    dist_match = re.search(r'H(\d+)m', header_line)
                    if dist_match:
                        races[race_num]['distance'] = int(dist_match.group(1))  # 数値で保存
                    
                    # 天気
                    weather_match = re.search(r'H\d+m\s+([晴曇雨])', header_line)
                    if weather_match:
                        races[race_num]['weather'] = weather_match.group(1)
                    
                    # 風向
                    wind_dir_match = re.search(r'風\s+([^\s\d]+?)[\s　]', header_line)
                    if wind_dir_match:
                        races[race_num]['wind_direction'] = wind_dir_match.group(1).replace('　', '').strip()
                    
                    # 風速
                    wind_speed_match = re.search(r'[\s　](\d+)m[\s　]+波', header_line)
                    if wind_speed_match:
                        races[race_num]['wind_speed'] = int(wind_speed_match.group(1))  # 数値で保存
                    
                    # 波高
                    wave_match = re.search(r'(\d+)cm', header_line)
                    if wave_match:
                        races[race_num]['wave_height'] = int(wave_match.group(1))  # 数値で保存
                    
                    break
                
                race_header_idx -= 1
            
            # 選手情報を抽出（全6艇を抽出、フライング含む）
            racer_idx = idx + 1
            
            # 区切り線をスキップ
            while racer_idx < len(section_lines) and '---' in section_lines[racer_idx]:
                racer_idx += 1
            
            # 選手情報を取得
            while racer_idx < len(section_lines):
                racer_line = section_lines[racer_idx]
                
                # 空行で終了
                if racer_line.strip() == '':
                    break
                
                # 行をsplit して、すべてのフィールドを抽出
                parts = racer_line.split()
                
                if len(parts) >= 11:  # 最小フィールド数
                    finish_order = parts[0]
                    boat_number = try_convert_number(parts[1])  # 数値変換
                    racer_id = try_convert_number(parts[2])  # 数値変換
                    
                    # 選手名は複数フィールド
                    racer_name_parts = []
                    j = 3
                    while j < len(parts) and not parts[j].isdigit():
                        racer_name_parts.append(parts[j])
                        j += 1
                    racer_name = ''.join(racer_name_parts)
                    
                    # 残りのフィールド
                    remaining = parts[j:] if j < len(parts) else []
                    
                    motor_age = try_convert_number(remaining[0] if len(remaining) > 0 else None)
                    boat_age = try_convert_number(remaining[1] if len(remaining) > 1 else None)
                    exhibition_time = try_convert_number(remaining[2] if len(remaining) > 2 else None)
                    gate_position = try_convert_number(remaining[3] if len(remaining) > 3 else None)
                    start_timing = try_convert_number(remaining[4] if len(remaining) > 4 else None)
                    race_time = remaining[5] if len(remaining) > 5 else None  # レースタイムは文字列のまま
                    
                    racer = {
                        'finish_order': finish_order,
                        'boat_number': boat_number,
                        'racer_id': racer_id,
                        'racer_name': racer_name,
                        'motor_age': motor_age,
                        'boat_age': boat_age,
                        'exhibition_time': exhibition_time,
                        'gate_position': gate_position,
                        'start_timing': start_timing,
                        'race_time': race_time
                    }
                    races[race_num]['racers'].append(racer)
                
                racer_idx += 1
            
            # 払戻金情報を抽出（2連単と3連単のみ）
            payout_idx = racer_idx + 1
            
            while payout_idx < len(section_lines):
                payout_line = section_lines[payout_idx].strip()
                
                # レース終了判定
                if payout_line == '' or re.match(r'\s*(\d+|[0-9]+)R\s+', payout_line):
                    break
                
                if not payout_line:
                    payout_idx += 1
                    continue
                
                # 2連単のみを抽出
                if '２連単' in payout_line or '2連単' in payout_line:
                    matches = re.findall(r'(\d+-\d+)\s+(\d+)', payout_line)
                    for match in matches:
                        payout_info = {
                            'type': '2連単',
                            'combination': match[0],
                            'payout': int(match[1]),  # 数値で保存
                            'odds': int(match[1]) / 100
                        }
                        races[race_num]['payouts'].append(payout_info)
                
                # 3連単のみを抽出
                elif '３連単' in payout_line or '3連単' in payout_line:
                    matches = re.findall(r'(\d+-\d+-\d+)\s+(\d+)', payout_line)
                    for match in matches:
                        payout_info = {
                            'type': '3連単',
                            'combination': match[0],
                            'payout': int(match[1]),  # 数値で保存
                            'odds': int(match[1]) / 100
                        }
                        races[race_num]['payouts'].append(payout_info)
                
                payout_idx += 1
    
    return {
        'venue': race_venue,
        'date': race_date,
        'day': race_day,
        'title': race_title,
        'races': races
    }

def create_venue_excel(venue_name, records):
    """レース場ごとのExcelワークブックを作成"""
    wb = Workbook()
    wb.remove(wb.active)
    
    # スタイル定義
    header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
    subheader_fill = PatternFill(start_color='8EAAD9', end_color='8EAAD9', fill_type='solid')
    header_font = Font(bold=True, color='FFFFFF', size=11)
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    center_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
    
    # 開催日でソート
    sorted_records = sorted(records, key=lambda x: x['date'])
    
    for record in sorted_records:
        date_str = record['date'].replace('/', '-')
        sheet_name = f"{date_str}"[:31]
        sheet = wb.create_sheet(title=sheet_name)
        
        # タイトル情報
        sheet['A1'] = f"レース場: {record['venue']}"
        sheet['A2'] = f"開催日: {record['date']} (第{record['day']}日)"
        sheet['A3'] = f"レースタイトル: {record['title']}"
        
        for row in [1, 2, 3]:
            sheet[f'A{row}'].font = Font(bold=True, size=12)
        
        current_row = 5
        
        # 各レース情報
        for race_num in sorted(record['races'].keys()):
            race = record['races'][race_num]
            
            # レース情報ヘッダー
            sheet[f'A{current_row}'] = f"【{race_num}R】"
            sheet[f'A{current_row}'].font = Font(bold=True, size=11, color='FFFFFF')
            sheet[f'A{current_row}'].fill = header_fill
            
            sheet[f'B{current_row}'] = race['type'] if race['type'] else ''
            
            # 距離は数値として保存
            if isinstance(race['distance'], int):
                sheet[f'C{current_row}'] = f"H{race['distance']}m"
            else:
                sheet[f'C{current_row}'] = race['distance'] if race['distance'] else ''
            
            sheet[f'D{current_row}'] = f"天気: {race['weather']}" if race['weather'] else "天気: -"
            
            # 風速は数値として保存
            if isinstance(race['wind_speed'], int):
                sheet[f'E{current_row}'] = f"風: {race['wind_direction']} {race['wind_speed']}m" if race['wind_direction'] else "風: -"
            else:
                sheet[f'E{current_row}'] = f"風: {race['wind_direction']} {race['wind_speed']}" if race['wind_direction'] else "風: -"
            
            # 波高は数値として保存
            if isinstance(race['wave_height'], int):
                sheet[f'F{current_row}'] = f"波: {race['wave_height']}cm"
            else:
                sheet[f'F{current_row}'] = f"波: {race['wave_height']}" if race['wave_height'] else "波: -"
            
            current_row += 1
            
            # 選手情報テーブル
            racer_headers = ['着', '艇', '登番', '選手名', 'モーター', 'ボート', 
                           '展示時間', '進入', 'ST', 'レースタイム']
            
            for col, header in enumerate(racer_headers, 1):
                cell = sheet.cell(row=current_row, column=col)
                cell.value = header
                cell.fill = subheader_fill
                cell.font = header_font
                cell.alignment = center_align
                cell.border = thin_border
            
            current_row += 1
            
            # 全6艇のデータを表示
            for racer in race['racers']:
                sheet.cell(row=current_row, column=1).value = racer['finish_order']
                sheet.cell(row=current_row, column=2).value = racer['boat_number']
                sheet.cell(row=current_row, column=3).value = racer['racer_id']
                sheet.cell(row=current_row, column=4).value = racer['racer_name']
                sheet.cell(row=current_row, column=5).value = racer['motor_age']
                sheet.cell(row=current_row, column=6).value = racer['boat_age']
                sheet.cell(row=current_row, column=7).value = racer['exhibition_time']
                sheet.cell(row=current_row, column=8).value = racer['gate_position']
                sheet.cell(row=current_row, column=9).value = racer['start_timing']
                sheet.cell(row=current_row, column=10).value = racer['race_time']
                
                # 数値フィールドの数値フォーマット
                for col in range(1, 11):
                    cell = sheet.cell(row=current_row, column=col)
                    cell.border = thin_border
                    cell.alignment = center_align
                    
                    # 数値はフォーマット設定
                    if col in [2, 3, 5, 6, 8]:  # 艇番、登番、モーター、ボート、進入
                        if isinstance(cell.value, int):
                            cell.number_format = '0'
                    elif col in [7, 9]:  # 展示時間、ST
                        if isinstance(cell.value, float):
                            cell.number_format = '0.00'
                
                current_row += 1
            
            current_row += 1
            
            # 払戻金情報（2連単と3連単のみ）
            if race['payouts']:
                payout_headers = ['投票種類', '組み合わせ', '払戻金', 'オッズ']
                for col, header in enumerate(payout_headers, 1):
                    cell = sheet.cell(row=current_row, column=col)
                    cell.value = header
                    cell.fill = subheader_fill
                    cell.font = header_font
                    cell.alignment = center_align
                    cell.border = thin_border
                
                current_row += 1
                
                for payout in race['payouts']:
                    cell_type = sheet.cell(row=current_row, column=1)
                    cell_type.value = payout['type']
                    
                    sheet.cell(row=current_row, column=2).value = payout['combination']
                    
                    cell_payout = sheet.cell(row=current_row, column=3)
                    cell_payout.value = payout['payout']
                    cell_payout.number_format = '#,##0'
                    
                    cell_odds = sheet.cell(row=current_row, column=4)
                    cell_odds.value = payout['odds']
                    cell_odds.number_format = '0.0'
                    
                    for col in range(1, 5):
                        sheet.cell(row=current_row, column=col).border = thin_border
                        sheet.cell(row=current_row, column=col).alignment = center_align
                    
                    current_row += 1
            
            current_row += 1
        
        # 列幅を自動調整
        sheet.column_dimensions['A'].width = 15
        sheet.column_dimensions['B'].width = 12
        sheet.column_dimensions['C'].width = 12
        sheet.column_dimensions['D'].width = 15
        sheet.column_dimensions['E'].width = 20
        sheet.column_dimensions['F'].width = 15
        for i in range(7, 11):
            sheet.column_dimensions[get_column_letter(i)].width = 12
    
    return wb

def process_yearly_data(base_dir, output_dir):
    """新フォルダ構造対応：YY年/MM月 形式のボートレースデータを処理"""
    
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # レース場ごとにデータを集約
    venue_data = {}
    
    print(f"入力ディレクトリ: {base_path}")
    print(f"出力ディレクトリ: {output_path}\n")
    
    # YY年 形式のフォルダを探す
    for year_folder in sorted(base_path.glob('*')):
        if not year_folder.is_dir():
            continue
        
        year_folder_name = year_folder.name
        # "25年", "22年" などのフォルダを検出
        year_match = re.match(r'(\d+)年', year_folder_name)
        if not year_match:
            continue
        
        year_str = year_match.group(1)
        year_int = int(year_str)
        
        # YY形式を西暦に変換
        if year_int <= 50:
            full_year = 2000 + year_int
        else:
            full_year = 1900 + year_int
        
        print(f"処理中: {year_folder_name} ({full_year}年)")
        
        # MM月 形式のサブフォルダを探す（01月、1月両対応）
        for month_folder in sorted(year_folder.glob('*')):
            if not month_folder.is_dir():
                continue
            
            month_folder_name = month_folder.name
            # "1月", "12月", "01月" などのフォルダを検出
            month_match = re.match(r'(\d+)月', month_folder_name)
            if not month_match:
                continue
            
            month_int = int(month_match.group(1))
            
            # TXTファイルを探す
            for txt_file in sorted(month_folder.glob('*.TXT')) + sorted(month_folder.glob('*.txt')):
                try:
                    # ファイルを読み込む
                    with open(txt_file, 'rb') as f:
                        content = f.read().decode('shift-jis')
                    
                    lines = content.split('\n')
                    
                    # セクションの境界（KBGN～KEND）を検出
                    section_starts = []
                    section_ends = []
                    
                    for i, line in enumerate(lines):
                        if re.match(r'\d+KBGN', line.strip()):
                            section_starts.append(i)
                        if re.match(r'\d+KEND', line.strip()):
                            section_ends.append(i)
                    
                    # 各セクションをパース
                    for sec_num, start_pos in enumerate(section_starts):
                        if sec_num >= len(section_ends):
                            break
                        
                        end_pos = section_ends[sec_num]
                        
                        # セクションをパース
                        data = parse_venue_section(lines, start_pos, end_pos + 1)
                        
                        if not data['venue'] or not data['races']:
                            continue
                        
                        venue = data['venue']
                        
                        # レース場ごとにデータを集約
                        if venue not in venue_data:
                            venue_data[venue] = []
                        
                        venue_data[venue].append(data)
                        
                        print(f"  ✓ {txt_file.name} → {venue} ({data['date']})")
                
                except Exception as e:
                    print(f"  ✗ {txt_file.name} エラー: {str(e)[:50]}")
    
    if not venue_data:
        print("\n❌ エラー: 有効なデータが見つかりません")
        return False
    
    # レース場ごとにExcelファイルを生成
    print(f"\n{'='*60}")
    print("Excel ファイルを生成中...\n")
    
    for venue, records in sorted(venue_data.items()):
        try:
            wb = create_venue_excel(venue, records)
            output_file = output_path / f"{venue}.xlsx"
            wb.save(str(output_file))
            print(f"✓ {venue}.xlsx ({len(records)} 日分)")
        except Exception as e:
            print(f"✗ {venue}.xlsx - エラー: {str(e)[:50]}")
    
    print(f"\n{'='*60}")
    print(f"✓ 完了: {output_path}")
    print(f"✓ 生成されたレース場数: {len(venue_data)}")
    return True

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # デフォルトパス
        base_dir = '/Users/maeshirotakao/ボートデータ/競走成績'
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = './競走成績_Excel'
    
    process_yearly_data(base_dir, output_dir)