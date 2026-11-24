# NSNG-Energy

Dựng lại phần thực nghiệm V của [CSA_Tung.pdf](docs/CSA_Tung.pdf) trên dữ liệu tiêu thụ điện của máy sấy. Quy trình huấn luyện bảy detector, kết hợp niềm tin bằng đồ thị thương lượng, rồi xuất các bảng, Figure 1 và dấu vết ra quyết định.

## Chạy lại

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m nsng_energy prepare
.venv\Scripts\python.exe -m nsng_energy run --device auto
```

Cần Python 3.11 trở lên. `--jobs 5` chạy năm fold trong các tiến trình song song (thời gian suy luận đo được khi đó bị ảnh hưởng bởi tranh chấp CPU). `python -m unittest discover -s tests` kiểm tra (18), (19), tính stochastic của ma trận cạnh và việc mang trust qua thời gian. `--device auto` chọn CUDA nếu bản PyTorch đã cài truy cập được GPU. Dùng `--device cpu` để chạy trên CPU. Nếu đã có ZIP của nguồn dữ liệu, truyền `prepare --archive <zip>`. Múi giờ trong dữ liệu được giữ theo bản ghi nguồn. `run.json` lưu tham số, phiên bản thư viện, nguồn dữ liệu và thời gian suy luận thực đo.

### Dùng dữ liệu và danh sách chuỗi riêng

Truyền `run --manifest <manifest.csv> --output <thư_mục>`. Manifest có các cột `sequence,path,group`. Các cột tùy chọn là `fault,normal_path,figure`. Đặt `figure=1` cho chuỗi cần vẽ. Các đường dẫn có thể tính từ thư mục của manifest. CSV chuỗi cần `ctime,activePower,label` với nhãn 0/1. Tên cột điện áp là `voltage` và cần cung cấp `--nominal-voltage` tính bằng volt. Tệp `normal_path` cung cấp dữ liệu hoạt động bình thường cho các detector khi nhóm tương ứng nằm trong tập huấn luyện. Nếu thiếu tệp đó, detector chỉ dùng các cửa sổ sạch thuộc tập huấn luyện.

Gán cùng `group` cho các bản lỗi xây dựng từ cùng phép đo hoặc có khoảng thu thập chồng lấn. Giao thức yêu cầu ít nhất ba nhóm thu thập và cả hai lớp trong tập huấn luyện và validation của từng fold. Có thể điều chỉnh `--folds` theo số nhóm.

## Nguồn và giao thức

Kho [Power Consumption Anomaly Dataset](https://github.com/AlexisPapaioannou/Power-Consumption-Anomaly-Dataset) đi kèm bài [Simulation of Malfunctions in Home Appliances’ Power Consumption](https://www.mdpi.com/1996-1073/17/17/4529). Bản commit được ghim trong mã và giữ thông tin giấy phép trong thư mục tải dữ liệu. Công suất bình thường là số liệu đo và các nhãn lỗi được tạo từ mô phỏng. Tài liệu [23] trong PDF trộn thông tin thư mục của bài mô hình hóa năng lượng của Pereira với số tập và mã bài của nghiên cứu công bố bộ dữ liệu. [Bài của Pereira và Martínez-Ramón](https://doi.org/10.1016/j.enbuild.2022.112653)

Nguồn công khai có 20 chuỗi thuộc hai loại lỗi paper nêu. Có 142 tập con 15 chuỗi khớp tổng số mẫu, hai đầu khoảng tỷ lệ lỗi làm tròn và chứa chuỗi `Faulty_Thermostat/day8` trong Figure 1. Mã chọn tổ hợp đầu tiên theo thứ tự từ điển của đường dẫn, trước khi tính bất kỳ metric mô hình nào. Bộ đã chọn có 15 chuỗi, 22.380 quan sát, 780 nhãn lỗi và tỷ lệ lỗi từ 2,04% đến 7,10%. `table_i_dataset.csv` và `splits.csv` ghi rõ tập dữ liệu và cách chia.

- **Detector.** Hai Isolation Forest với contamination 0,05 và 0,10, hai One-Class SVM với nu 0,05 và 0,10, LOF ở chế độ novelty, LSTM autoencoder và Adaptive Z Score. Các đặc trưng và cửa sổ dùng lịch sử về trước. Detector học trên phần bình thường của nhóm train. Điểm thô được ánh xạ sigmoid với tâm và thang đo định từ train, confidence tăng theo khoảng cách đến điểm 0,5.
- **Đồ thị thương lượng (Algorithm 1).** Các quan sát của một chuỗi được thương lượng tuần tự theo thời gian; các chuỗi độc lập chạy song song trong một batch có padding. Với mỗi quan sát, đồ thị được dựng một lần theo (4)–(7) từ niềm tin ban đầu (dòng 2), rồi lặp tối đa K vòng GAT (10)–(13) và quy tắc (14)–(17). Trust τ được mang từ quan sát t sang t+1 theo (9) và khởi tạo 0,8 ở đầu mỗi chuỗi. Ngưỡng xung đột δ_ij của (2) dùng confidence của vòng hiện tại với γ = 0,5, λ = 1. Xung đột được tách thành direct contradiction (một điểm ≥ 0,7, điểm kia ≤ 0,3, mục III.C) và ambiguous divergence. Consensus dùng (18) và (19), trong đó độ đồng thuận là `1 − |s_i − s_*|/2`. Giữ tối đa 5 cạnh đến khác self trước khi thêm các cạnh R3. Một agent không có cạnh trao đổi đi ra giữ trọng số ở self để mỗi hàng tổng bằng 1. Dùng EMA 0,95 cho κ_ij và d_ij, GAT 4 đầu 16 chiều, cùng một bộ tham số qua các vòng. Lớp dự đoán theo đúng (13); `--detector-residual w` cộng thêm `w·logit(s_i^(0))` nếu cần ổn định huấn luyện. GAT học bằng truncated BPTT theo thời gian (`--chunk-size` bước, mặc định 32) với BCE có trọng số lớp và thành phần hiệu chỉnh confidence. Mặc định dùng seed 42, cửa sổ 24, 8 epoch LSTM và 4 epoch GAT. Paper không cung cấp các lựa chọn huấn luyện và tiền xử lý này.
- **Quy tắc.** R1 dùng hiệu chỉnh mềm theo phương trình (16), với tùy chọn `--voltage-rule hard` áp dụng sàn 0,8 theo Table II. R2 so sánh điểm của agent ở vòng k tại thời điểm t với vòng k tại t−1 và đặt confidence bằng nửa confidence của t−1 khi độ lệch vượt 0,3 (Table II). R3 dùng Pearson trên tối đa 32 điểm quá khứ của từng cặp detector và đánh dấu xung đột khi các điều kiện correlation và độ lệch thỏa. R4 nhân trust với 0,9 khi một agent có hơn 5 xung đột trong vòng hiện tại; vì trust được mang theo thời gian nên hiệu ứng của R4 kéo dài sang các quan sát sau. Dữ liệu dryer công khai không chứa số đo điện áp nên R1 không kích hoạt.
- **Đánh giá.** Dùng tối đa 12 vòng, ngưỡng dừng 0,0001, trust momentum 0,9 và symbolic strength 0,4. Mỗi fold dùng các nhóm thu thập riêng cho train, validation và test. Chọn ngưỡng theo F1 trên validation trong lưới 0,30–0,70, bước 0,05. Gán nhãn bất thường nếu điểm consensus lớn hơn ngưỡng, theo mục IV.F. Hai baseline tổng hợp dùng cùng công thức trung bình theo confidence như mô tả ở mục V.B. Báo cáo metric gộp trên toàn bộ dự đoán test và trung bình theo chuỗi. Kiểm định hoán vị hai phía đánh giá chênh lệch F1 theo cặp nhóm thu thập.
- **Ablation và sensitivity.** Table IV tắt từng quy tắc tại suy luận trên cùng checkpoint trong fold. Table V huấn luyện lại khi bỏ LSTM, trust động, toàn bộ quy tắc hoặc đổi sang attention một đầu với cùng chiều rộng. Sensitivity dùng checkpoint đầy đủ để quét số vòng, trust momentum và hệ số hiệu chỉnh mềm của R1. Từng thiết lập chọn ngưỡng trên validation. `threshold_grid_test.csv` và `best_test_grid.csv` ghi thêm kết quả quét trực tiếp trên test để xem ảnh hưởng của cách tối ưu ngưỡng mô tả trong paper. Giá trị tốt nhất trong lưới test là đánh giá hồi cứu và có thể lạc quan.

## Đọc kết quả

Lệnh `run` ghi kết quả vào `results/public-dryer/` theo mặc định. Các tệp `predictions.csv.gz`, `consensus_kg.jsonl.gz` và `traces/` được xuất khi chạy. JSONL liên kết quan sát với điểm, confidence, dự đoán, số vòng, số xung đột (tổng và direct contradiction) trước/sau thương lượng, trust cuối của từng agent và dấu vết của assessment. Điểm của cả chuỗi là điểm tối đa của các quan sát theo (20). Mỗi NPZ chứa mảng `states[round, sample, agent, property]`, `outgoing[round, sample, source, target]`, `conflict_edges` và `rule_hits[round, sample, agent, rule]`. `states[0]` là niềm tin ban đầu. `rounds[sample]` cho biết số vòng thực sự dùng cho quan sát đó.

| Đầu ra | Nội dung |
| --- | --- |
| `table_i_dataset.csv`, `splits.csv` | Dữ liệu thực nghiệm và các nhóm của từng fold |
| `table_iii_methods.csv`, `per_sequence.csv` | NSNG-Energy, hai baseline và bảy agent |
| `table_iv_rules.csv`, `table_v_components.csv` | Ablation quy tắc và thành phần |
| `sensitivity.csv`, `threshold_grid_test.csv`, `best_test_grid.csv` | Quét siêu tham số và ngưỡng |
| `figure_1.png`, `figure_1_metrics.json` | Công suất, nhãn, điểm từng agent, consensus và ma trận nhầm lẫn của 1.000 mẫu đầu ở ngưỡng 0,45 |
| `paired_significance.json`, `run.json` | Kiểm định thống kê, thông tin lần chạy và thời gian suy luận theo batch |
| `paper_comparison.csv` | Mỗi số công bố trong Table I, III, IV, V, mục V.I, Figure 1 và Table VI đặt cạnh kết quả tái dựng: metric gộp, trung bình theo chuỗi và ngưỡng tốt nhất trên test (giao thức của paper) |
| `table_vi_qualitative.csv` | Thông tin định tính AutoGen/MetaGPT được ghi trong PDF. Paper không có thử nghiệm đối chứng cùng dữ liệu cho hai hệ này. |

### Mức độ đối chiếu với paper

Paper báo NSNG-Energy đạt Accuracy 0,8113, Precision 0,1679, Recall 0,5625, F1 0,2379 và ROC-AUC 0,7837. Nếu precision và recall gộp từ một ma trận nhầm lẫn, chúng cho F1 khoảng 0,2586. Paper chưa mô tả cách gộp các metric. Figure 1 ghi TN 775, FP 158, FN 67, TP 0. Các ô đó cho Accuracy 0,775 và F1 0, trong khi tiêu đề ghi 0,811 và 0,238. Figure được dựng lại tính các metric trực tiếp trên mẫu hiển thị và ma trận thực đo. Table IV và Table V cũng đưa ra hai F1 khác nhau cho việc bỏ toàn bộ quy tắc. Hai giao thức ablation của dự án được ghi rõ ở trên.

Các metric tái dựng phụ thuộc vào danh sách chuỗi, cách chia, tiền xử lý và quy trình học giả định. Việc xác nhận chính xác các số công bố cần danh sách tệp, cấu hình hoặc mã gốc và cách tổng hợp của tác giả.
