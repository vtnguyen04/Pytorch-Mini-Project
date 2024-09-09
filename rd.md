### MobileNetV1: Giới Thiệu

**MobileNetV1** là một kiến trúc mạng nơ-ron tích chập sâu (CNN) được thiết kế đặc biệt cho các thiết bị di động và nhúng, nơi mà tài nguyên tính toán như CPU, bộ nhớ và năng lượng đều bị hạn chế. MobileNetV1 được giới thiệu lần đầu tiên trong bài báo **"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"** bởi nhóm nghiên cứu của Google vào năm 2017.

### Động Lực Phát Triển MobileNetV1

CNN truyền thống như **VGG** hoặc **ResNet** có hiệu suất rất tốt nhưng lại yêu cầu tài nguyên tính toán và bộ nhớ lớn, điều này không phù hợp khi triển khai trên các thiết bị có tài nguyên hạn chế như điện thoại di động, thiết bị IoT, hoặc các hệ thống nhúng. Để giải quyết vấn đề này, nhóm nghiên cứu Google đã phát triển MobileNetV1 với mục tiêu:

- **Giảm thiểu số lượng tham số và phép tính toán** nhưng vẫn duy trì hiệu suất nhận dạng cao.
- **Tối ưu hóa tốc độ tính toán và tiêu thụ tài nguyên** khi triển khai trên các thiết bị có giới hạn về tài nguyên.

### Kiến Trúc MobileNetV1

MobileNetV1 sử dụng một kỹ thuật mới được gọi là **Depthwise Separable Convolutions** để thay thế cho các tầng tích chập chuẩn (standard convolutions). Mục tiêu chính của kiến trúc này là giảm đáng kể số lượng phép tính và tham số trong mạng.

#### Depthwise Separable Convolutions

Để hiểu rõ hơn về cải tiến của MobileNetV1, ta cần hiểu về **Depthwise Separable Convolutions**, gồm hai bước chính:

1. **Depthwise Convolution**: Thay vì áp dụng tích chập chuẩn với cùng một bộ lọc trên tất cả các kênh của đầu vào, MobileNetV1 áp dụng một bộ lọc riêng biệt cho mỗi kênh đầu vào. Điều này giúp giảm đáng kể số lượng phép tính.
   
2. **Pointwise Convolution**: Sau khi áp dụng Depthwise Convolution, Pointwise Convolution áp dụng tích chập 1x1 để kết hợp các kênh đầu ra. Mặc dù đây là một phép tính đơn giản, nó giúp giảm thiểu số lượng tham số trong mô hình.

Mô hình tích chập chuẩn cần thực hiện phép tính trên tất cả các kênh đầu vào cùng một lúc, trong khi Depthwise Separable Convolutions chia quá trình này thành hai bước: đầu tiên xử lý từng kênh riêng lẻ, sau đó kết hợp chúng lại. Điều này giúp giảm bớt số lượng phép tính và tham số của mô hình.

#### Toán Học Của Depthwise Separable Convolutions

- Với một phép tích chập chuẩn, số lượng phép tính cần thiết là:
  \[
  D_K \times D_K \times M \times N \times D_F \times D_F
  \]
  Trong đó:
  - \(D_K \times D_K\) là kích thước của kernel (bộ lọc).
  - \(M\) là số kênh đầu vào.
  - \(N\) là số kênh đầu ra.
  - \(D_F \times D_F\) là kích thước của đầu ra.

- Với Depthwise Separable Convolution, số lượng phép tính cần thiết được chia làm hai phần:
  - Depthwise Convolution: \(D_K \times D_K \times M \times D_F \times D_F\).
  - Pointwise Convolution: \(1 \times 1 \times M \times N \times D_F \times D_F\).
  
  Tổng số lượng phép tính cần thiết:
  \[
  D_K \times D_K \times M \times D_F \times D_F + M \times N \times D_F \times D_F
  \]
  So với tích chập chuẩn, Depthwise Separable Convolutions giảm số lượng phép tính khoảng 8 đến 9 lần khi sử dụng kernel 3x3.

### Các Cải Tiến Chính Của MobileNetV1

1. **Depthwise Separable Convolutions**: Đây là cải tiến chính giúp giảm số lượng phép tính toán và tham số một cách đáng kể.
   
2. **Sử dụng hai tham số điều chỉnh (hyperparameters)**:
   - **Width Multiplier (\(\alpha\))**: Điều chỉnh số lượng kênh đầu ra của mỗi lớp. Nếu giảm \(\alpha\), số lượng kênh đầu ra sẽ giảm, dẫn đến giảm số lượng phép tính và tham số.
   - **Resolution Multiplier (\(\rho\))**: Điều chỉnh độ phân giải của đầu vào. Giảm độ phân giải đầu vào cũng giúp giảm số lượng phép tính.

### Ưu Điểm Của MobileNetV1

- **Hiệu quả tính toán cao**: Nhờ sử dụng Depthwise Separable Convolutions, MobileNetV1 giảm đáng kể số lượng phép tính toán và tham số so với các mô hình CNN truyền thống.
  
- **Thích hợp cho thiết bị di động và nhúng**: MobileNetV1 được thiết kế để tối ưu hóa tốc độ và giảm tiêu thụ tài nguyên, rất phù hợp cho các thiết bị có tài nguyên hạn chế.

- **Có thể điều chỉnh**: MobileNetV1 cung cấp các tham số điều chỉnh như Width Multiplier và Resolution Multiplier giúp người dùng cân bằng giữa độ chính xác và tốc độ phù hợp với từng ứng dụng cụ thể.

### Nhược Điểm Của MobileNetV1

- **Hiệu suất thấp hơn so với các mạng sâu hơn**: Mặc dù MobileNetV1 có hiệu suất cao về mặt tính toán, độ chính xác của nó vẫn kém hơn so với các mô hình lớn hơn như ResNet hay Inception.

- **Không phải là tối ưu cho tất cả các tác vụ**: Mặc dù MobileNetV1 rất tốt cho các tác vụ nhận dạng hình ảnh trên thiết bị di động, nó có thể không tối ưu cho một số nhiệm vụ yêu cầu độ chính xác cao hơn.

### Tóm Tắt

- **MobileNetV1** là một CNN nhẹ, dựa trên Depthwise Separable Convolution để giảm số lượng phép tính và tham số.
- **Động lực chính** của việc phát triển MobileNetV1 là để tạo ra một mô hình có thể chạy hiệu quả trên các thiết bị di động với tài nguyên hạn chế.
- **Ưu điểm** của MobileNetV1 bao gồm hiệu quả tính toán cao, khả năng điều chỉnh linh hoạt và phù hợp cho các ứng dụng nhúng.
- **Nhược điểm** của nó là hiệu suất nhận dạng thấp hơn so với các mạng lớn hơn, và không phù hợp cho tất cả các loại tác vụ đòi hỏi độ chính xác cao.

MobileNetV1 mở đầu cho một loạt các kiến trúc MobileNet tiếp theo (MobileNetV2, MobileNetV3) với các cải tiến hơn nữa về hiệu quả và độ chính xác.

### MobileNetV2: Giới Thiệu

**MobileNetV2** là phiên bản cải tiến của MobileNetV1, được giới thiệu trong bài báo **"MobileNetV2: Inverted Residuals and Linear Bottlenecks"** vào năm 2018 bởi nhóm nghiên cứu của Google. MobileNetV2 tiếp tục tập trung vào việc tối ưu hóa kiến trúc CNN để giảm thiểu tính toán và tham số, nhưng đồng thời cải thiện độ chính xác của mô hình so với phiên bản trước.

### Động Lực Phát Triển MobileNetV2

Mặc dù MobileNetV1 đã rất thành công trong việc giảm số lượng phép tính toán và tham số, nhưng vẫn có không gian để tối ưu thêm về mặt hiệu suất. Động lực cho việc phát triển MobileNetV2 đến từ một số vấn đề sau:

- **Khả năng thiếu hiệu quả trong việc trích xuất đặc trưng**: MobileNetV1 sử dụng Depthwise Separable Convolutions để giảm tính toán, nhưng việc xử lý thông tin ở các tầng với độ phân giải cao vẫn cần được tối ưu thêm.
  
- **Cải thiện hiệu suất mà không tăng quá nhiều độ phức tạp**: Để cạnh tranh với các kiến trúc CNN mạnh mẽ hơn như ResNet và DenseNet, nhưng vẫn giữ nguyên lợi thế về tính toán nhẹ.

### Các Cải Tiến Chính Trong MobileNetV2

MobileNetV2 giới thiệu hai cải tiến kiến trúc chính:

1. **Inverted Residuals (Còn Gọi Là Block Bottleneck Đảo Ngược)**:
   - Trong ResNet, các khối residual (residual blocks) sử dụng cấu trúc "bottleneck" để giảm số lượng tham số bằng cách giảm số chiều của không gian đặc trưng (feature space), sau đó mở rộng không gian đặc trưng trở lại. Tuy nhiên, MobileNetV2 đi theo hướng ngược lại.
   - **Inverted Residual Block**: Thay vì giảm số chiều trước rồi mở rộng, MobileNetV2 bắt đầu với một không gian đặc trưng có số chiều nhỏ (bottleneck) và mở rộng nó ra, sau đó lại giảm số chiều bằng pointwise convolution (1x1 convolution). Điều này cho phép mô hình giữ lại nhiều thông tin hơn trong quá trình trích xuất đặc trưng mà không tăng quá nhiều số phép tính toán.

2. **Linear Bottlenecks**:
   - Ở cuối mỗi Inverted Residual Block, MobileNetV2 sử dụng một tầng **linear bottleneck** thay vì hàm phi tuyến như ReLU. Điều này quan trọng vì các nghiên cứu đã chỉ ra rằng việc áp dụng hàm phi tuyến trong không gian có số chiều thấp có thể làm mất thông tin quan trọng. Linear bottleneck giúp duy trì thông tin đặc trưng một cách hiệu quả hơn.

#### Kiến Trúc Của Inverted Residual Block

Mỗi **Inverted Residual Block** trong MobileNetV2 bao gồm các bước sau:

1. **Expansion**: Mở rộng chiều không gian đặc trưng bằng tích chập 1x1 (pointwise convolution). Thông thường, số chiều không gian đặc trưng được mở rộng theo hệ số 6 (\(t = 6\)).
   
2. **Depthwise Convolution**: Áp dụng tích chập depthwise để trích xuất đặc trưng không gian mà không tăng số lượng kênh.

3. **Linear Bottleneck**: Cuối cùng, một tích chập 1x1 (pointwise convolution) được áp dụng để giảm số chiều không gian đặc trưng về lại số chiều ban đầu.

Ngoài ra, nếu đầu vào và đầu ra của block có cùng kích thước, một **kết nối tắt (skip connection)** sẽ được sử dụng, tương tự như trong ResNet.

### Toán Học Của MobileNetV2

Mỗi Inverted Residual Block có thể được biểu diễn qua các bước toán học sau:

- Giả sử đầu vào có số chiều \(D_{in}\), bước đầu tiên là mở rộng không gian đặc trưng lên \(t \times D_{in}\) bằng pointwise convolution với kernel 1x1.
  
- Tiếp theo, thực hiện Depthwise Convolution trên không gian mở rộng đó mà không thay đổi số chiều kênh.

- Cuối cùng, sử dụng pointwise convolution để giảm không gian đặc trưng về lại \(D_{out}\), chiều không gian ban đầu.

Nếu \(D_{in} = D_{out}\), thì một kết nối tắt sẽ được thêm vào để kết hợp đầu vào và đầu ra.

### Hiệu Quả Của MobileNetV2

So với MobileNetV1, MobileNetV2 không chỉ giảm số phép tính toán trong các tầng tích chập mà còn cải thiện hiệu suất mô hình nhờ các cải tiến sau:

1. **Inverted Residual Block** giúp duy trì nhiều thông tin hơn trong quá trình trích xuất đặc trưng.
   
2. **Linear Bottleneck** giúp tránh mất thông tin khi áp dụng hàm phi tuyến trong không gian có số chiều thấp.

### So Sánh MobileNetV1 Và MobileNetV2

| Đặc Điểm                | MobileNetV1                              | MobileNetV2                                          |
|-------------------------|------------------------------------------|-----------------------------------------------------|
| **Tích chập chính**      | Depthwise Separable Convolutions         | Inverted Residuals + Depthwise Separable Convolutions |
| **Bottleneck**           | Không có bottleneck                      | Có Linear Bottleneck                                 |
| **Kết nối tắt (skip)**   | Không                                    | Có (khi đầu vào và đầu ra cùng kích thước)           |
| **Hiệu suất**            | Tốt cho các thiết bị di động             | Hiệu suất cao hơn nhờ bảo toàn thông tin tốt hơn     |
| **Số phép tính toán**    | Ít hơn so với CNN truyền thống           | Ít hơn MobileNetV1 ở các tầng tích chập sâu hơn      |
| **Độ chính xác**         | Tốt nhưng kém hơn so với các mô hình lớn | Cải thiện đáng kể so với MobileNetV1                 |

### Ưu Điểm Của MobileNetV2

- **Hiệu suất cao hơn so với MobileNetV1**: Nhờ vào Inverted Residuals và Linear Bottlenecks, MobileNetV2 đạt được độ chính xác cao hơn mà không tăng quá nhiều độ phức tạp.
  
- **Giữ lại nhiều thông tin hơn**: Sử dụng kết nối tắt cùng với Linear Bottlenecks giúp MobileNetV2 duy trì thông tin đặc trưng tốt hơn trong quá trình huấn luyện.

- **Tiêu thụ tài nguyên thấp hơn**: Mặc dù MobileNetV2 phức tạp hơn một chút so với MobileNetV1, nhưng nó vẫn duy trì được hiệu quả tính toán cao và phù hợp cho các thiết bị di động và nhúng.

### Nhược Điểm Của MobileNetV2

- **Phức tạp hơn so với MobileNetV1**: Mặc dù hiệu suất được cải thiện, nhưng MobileNetV2 có kiến trúc phức tạp hơn, đòi hỏi kỹ thuật triển khai tinh vi hơn trên các thiết bị có tài nguyên rất hạn chế.

- **Cần thêm tài nguyên**: Dù MobileNetV2 vẫn rất hiệu quả, nhưng so với MobileNetV1, nó yêu cầu tài nguyên tính toán và bộ nhớ cao hơn một chút.

### Tóm Tắt

- **MobileNetV2** là phiên bản cải tiến của MobileNetV1, với các khối **Inverted Residuals** và **Linear Bottlenecks** giúp cải thiện hiệu suất mà vẫn duy trì tính hiệu quả về mặt tính toán.
- **Inverted Residual Block** là cải tiến chính, giúp giữ lại thông tin và giảm thiểu số phép tính toán không cần thiết.
- MobileNetV2 phù hợp cho các ứng dụng trên thiết bị di động và nhúng, nơi mà tài nguyên tính toán bị hạn chế nhưng vẫn cần độ chính xác cao.
- **Ưu điểm** của MobileNetV2 là hiệu suất cao hơn so với MobileNetV1, trong khi **nhược điểm** là độ phức tạp cao hơn và yêu cầu tài nguyên tăng lên đôi chút.

MobileNetV2 là bước tiến lớn so với MobileNetV1 và là một trong những kiến trúc CNN nhẹ tốt nhất cho các thiết bị di động vào thời điểm nó được ra mắt.

### MobileNetV3: Giới Thiệu

**MobileNetV3** là phiên bản cải tiến tiếp theo của dòng kiến trúc MobileNet, được giới thiệu trong bài báo **"Searching for MobileNetV3"** vào năm 2019. MobileNetV3 kết hợp những cải tiến từ **MobileNetV2** với các kỹ thuật tối ưu hóa tiên tiến như **Neural Architecture Search (NAS)** để tự động thiết kế và tinh chỉnh kiến trúc mạng, nhằm đạt hiệu suất cao hơn nữa trên các thiết bị di động.

### Động Lực Phát Triển MobileNetV3

MobileNetV2 đã đạt được mức cân bằng tốt giữa độ chính xác và hiệu quả tính toán, nhưng với sự phát triển không ngừng của các ứng dụng yêu cầu xử lý trên thiết bị di động, nhóm nghiên cứu MobileNetV3 đã cải tiến thêm dựa trên:

- **Tối ưu hóa kiến trúc**: Các mô hình CNN nhẹ cần phải tiếp tục cải thiện về cả tốc độ và độ chính xác mà không làm tăng quá nhiều độ phức tạp.
  
- **Sử dụng kỹ thuật tự động tìm kiếm kiến trúc**: Neural Architecture Search (NAS) giúp tìm ra các cấu trúc tốt nhất cho các tác vụ cụ thể, thay vì dựa vào thiết kế thủ công như trước.

- **Tối ưu hóa các thành phần**: Kết hợp các kỹ thuật tối ưu hóa mới như **SE (Squeeze-and-Excitation)** để cải thiện khả năng trích xuất đặc trưng mà vẫn giữ được hiệu quả tính toán.

### Các Cải Tiến Chính Trong MobileNetV3

MobileNetV3 kết hợp nhiều cải tiến từ MobileNetV2 và thêm một số kỹ thuật mới:

1. **Neural Architecture Search (NAS)**:
   - MobileNetV3 sử dụng NAS để tự động tìm kiếm kiến trúc mạng tối ưu cho các tác vụ cụ thể. Thay vì thiết kế thủ công như MobileNetV1 và V2, NAS giúp tìm ra các khối tích chập phù hợp nhất với từng tầng của mạng.
   - Nhờ NAS, MobileNetV3 có cấu trúc tối ưu hơn về mặt tính toán mà vẫn duy trì được độ chính xác cao.

2. **Squeeze-and-Excitation (SE) Blocks**:
   - MobileNetV3 tích hợp **Squeeze-and-Excitation (SE) blocks**, một kỹ thuật đã chứng minh hiệu quả trong việc tăng cường chất lượng trích xuất đặc trưng bằng cách học cách "chú ý" đến các kênh đầu vào quan trọng.
   - SE block hoạt động bằng cách nén không gian đặc trưng (squeeze) và sau đó "khuếch đại" (excite) các kênh quan trọng, giúp mô hình tập trung vào các đặc trưng cần thiết.

3. **Hard-Swish Activation**:
   - Kích hoạt chuẩn **ReLU** trong MobileNetV2 được thay thế bằng **Hard-Swish**, một phiên bản đơn giản hóa của hàm **Swish** (hoạt động tương tự như ReLU nhưng có độ trơn cao hơn, giúp tăng hiệu suất). Hard-Swish có lợi thế tính toán nhanh và cung cấp đặc tính phi tuyến tính tốt hơn.
   - Công thức của **Hard-Swish**:
     \[
     \text{Hard-Swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}
     \]
     So với Swish, Hard-Swish có thể tính toán nhanh hơn và hiệu quả hơn trên các thiết bị di động.

4. **Inverted Residuals từ MobileNetV2**:
   - **Inverted Residual Block** từ MobileNetV2 vẫn được giữ lại trong MobileNetV3, vì đây là yếu tố giúp mạng hoạt động hiệu quả trên các thiết bị có tài nguyên hạn chế.
   - Tuy nhiên, các khối này được tinh chỉnh và kết hợp với SE block để tăng khả năng trích xuất đặc trưng.

### Kiến Trúc MobileNetV3: Large và Small

MobileNetV3 có hai biến thể chính:

1. **MobileNetV3-Large**: 
   - Được tối ưu hóa cho các tác vụ yêu cầu độ chính xác cao hơn, nhưng vẫn tiết kiệm tài nguyên. Đây là biến thể phù hợp cho các ứng dụng yêu cầu xử lý phức tạp.
   
2. **MobileNetV3-Small**: 
   - Được tối ưu hóa cho các tác vụ yêu cầu tính toán nhanh hơn, tiêu thụ ít tài nguyên hơn, nhưng độ chính xác có thể thấp hơn so với bản Large. Đây là biến thể rất phù hợp cho các thiết bị cực kỳ hạn chế về tài nguyên.

#### Cấu Trúc Chi Tiết Của MobileNetV3

**MobileNetV3-Large** bao gồm tổng cộng 16 tầng tích chập (convolutional layers) và sử dụng nhiều khối Inverted Residual với các SE block và Hard-Swish activation. Tương tự, **MobileNetV3-Small** sử dụng 11 tầng tích chập nhưng được tinh chỉnh để giảm bớt số lượng phép tính toán.

### So Sánh MobileNetV1, V2 và V3

| Đặc Điểm                  | MobileNetV1                               | MobileNetV2                                          | MobileNetV3                                  |
|---------------------------|-------------------------------------------|-----------------------------------------------------|----------------------------------------------|
| **Tích chập chính**        | Depthwise Separable Convolutions          | Inverted Residuals + Depthwise Separable Convolutions | Inverted Residuals + SE Blocks               |
| **Kích hoạt**              | ReLU                                      | ReLU6                                               | Hard-Swish                                   |
| **Bottleneck**             | Không có bottleneck                       | Linear Bottleneck                                   | Linear Bottleneck + SE Block                 |
| **Tối ưu hóa kiến trúc**   | Thiết kế thủ công                         | Thiết kế thủ công                                   | Neural Architecture Search (NAS)             |
| **Kết nối tắt (skip)**     | Không                                     | Có (nếu \(D_{in} = D_{out}\))                       | Có (tinh chỉnh với SE block)                |
| **Hiệu suất**              | Tiết kiệm tài nguyên nhưng độ chính xác thấp | Hiệu suất tốt hơn với độ chính xác tăng đáng kể      | Hiệu suất cao nhất trong dòng MobileNet      |
| **Số lượng phép tính toán**| Ít hơn các CNN truyền thống               | Ít hơn MobileNetV1                                  | Ít hơn MobileNetV2 với hiệu suất cao hơn     |

### Ưu Điểm Của MobileNetV3

1. **Hiệu suất tốt hơn MobileNetV2**: Nhờ vào NAS và SE blocks, MobileNetV3 đạt được độ chính xác cao hơn mà vẫn duy trì được hiệu quả tính toán tốt.

2. **Tối ưu hóa tốt hơn cho thiết bị di động**: Các kỹ thuật như Hard-Swish và SE block giúp MobileNetV3 hoạt động nhanh hơn trên các thiết bị di động mà không tiêu tốn quá nhiều tài nguyên.

3. **Kiến trúc tự động hóa**: Sử dụng NAS giúp MobileNetV3 được tối ưu hóa tự động cho các tác vụ cụ thể, giúp tối đa hóa hiệu suất mà không cần quá nhiều can thiệp thủ công.

4. **Hai phiên bản (Large và Small)**: MobileNetV3 cung cấp hai biến thể cho các yêu cầu khác nhau, giúp linh hoạt hơn trong việc triển khai vào các ứng dụng thực tế.

### Nhược Điểm Của MobileNetV3

1. **Phức tạp hơn**: Dù NAS đã tối ưu hóa kiến trúc, nhưng MobileNetV3 vẫn phức tạp hơn so với MobileNetV2, đòi hỏi nhiều hơn về kỹ thuật triển khai, đặc biệt khi tích hợp SE blocks và Hard-Swish.

2. **Yêu cầu tài nguyên cao hơn MobileNetV2**: Mặc dù hiệu quả được tăng lên, nhưng MobileNetV3 yêu cầu tài nguyên cao hơn đôi chút so với MobileNetV2, đặc biệt là khi sử dụng phiên bản Large.

### Tóm Tắt

- **MobileNetV3** là phiên bản hiện đại và tối ưu nhất trong dòng MobileNet, kết hợp các kỹ thuật tự động tìm kiếm kiến trúc (NAS) và các phương pháp tối ưu hóa mới như SE blocks và Hard-Swish để cải thiện hiệu suất trên các thiết bị di động.
- **Inverted Residual Block** từ MobileNetV2 vẫn được giữ lại và cải tiến thêm bằng SE blocks để tăng khả năng trích xuất đặc trưng.
- **Ưu điểm** chính của MobileNetV3 là hiệu suất cao hơn, độ chính xác tốt hơn, và khả năng tối ưu hóa tốt cho các thiết bị di động.
- **Nhược điểm** là độ phức tạp cao hơn và đôi khi yêu cầu tài nguyên cao hơn so với MobileNetV2, nhưng vẫn rất phù hợp cho các ứng dụng di động và nhúng.

MobileNetV3 là kiến trúc CNN nhẹ tiên tiến nhất trong dòng MobileNet và đã đạt được sự cân bằng tuyệt vời giữa độ chính xác và hiệu quả tính toán, phù hợp cho các ứng dụng yêu cầu xử lý nhanh nhưng vẫn đảm bảo hiệu suất trên các thiết bị có tài nguyên hạn chế.

### Neural Architecture Search (NAS): Giới Thiệu

**Neural Architecture Search (NAS)** là một phương pháp tự động tìm kiếm kiến trúc mạng nơ-ron tối ưu cho một tác vụ cụ thể, thay vì dựa vào việc thiết kế thủ công bởi các chuyên gia. NAS sử dụng các thuật toán học máy để khám phá không gian kiến trúc mạng nơ-ron và tìm ra các mô hình tốt nhất về cả độ chính xác và hiệu quả tính toán.

### Động Lực Phát Triển NAS

Trong nhiều năm, các kiến trúc mạng nơ-ron (như **AlexNet**, **VGG**, **ResNet**, và **Inception**) phần lớn được thiết kế thủ công bởi các nhà nghiên cứu. Tuy nhiên, việc thiết kế thủ công có một số hạn chế:

1. **Rất tốn thời gian và công sức**: Thiết kế thủ công một kiến trúc CNN tốt đòi hỏi rất nhiều thử nghiệm và điều chỉnh, thường mất hàng tuần hoặc hàng tháng.
   
2. **Không đảm bảo tối ưu**: Ngay cả với các chuyên gia, việc tìm ra một kiến trúc mạng tối ưu cho một tác vụ cụ thể không phải lúc nào cũng dễ dàng và có thể dẫn đến những kiến trúc không đạt hiệu suất cao nhất.

3. **Sự phức tạp của không gian tìm kiếm**: Không gian kiến trúc mạng nơ-ron rất lớn với nhiều yếu tố như số lớp, kích thước lớp, loại lớp, hàm kích hoạt, kết nối giữa các lớp, v.v. Điều này khiến việc tìm kiếm thủ công gặp nhiều khó khăn.

Để giải quyết các vấn đề này, **NAS** ra đời với mục tiêu tự động hóa quá trình tìm kiếm kiến trúc mạng nơ-ron tối ưu, nhằm giảm thiểu sự can thiệp thủ công và tìm ra các mô hình có hiệu suất tốt nhất.

### Cách Thức Hoạt Động Của NAS

NAS là một quá trình tìm kiếm trong không gian kiến trúc mạng nơ-ron. Quá trình này có thể được chia thành ba thành phần chính:

1. **Không gian tìm kiếm (Search Space)**:
   - Đây là tập hợp tất cả các kiến trúc mạng nơ-ron có thể được tạo ra. Không gian tìm kiếm có thể bao gồm các kiến trúc khác nhau về số lớp, loại lớp (tích chập, fully connected, pooling), kích thước lớp, hàm kích hoạt, kết nối tắt (skip connections), v.v.
   - Không gian tìm kiếm được thiết kế sao cho đủ linh hoạt để chứa nhiều loại kiến trúc, nhưng cũng cần đủ hẹp để quá trình tìm kiếm không mất quá nhiều thời gian.

2. **Chiến lược tìm kiếm (Search Strategy)**:
   - Đây là cách mà NAS duyệt qua không gian tìm kiếm để tìm ra kiến trúc mạng nơ-ron tốt nhất. Có nhiều phương pháp khác nhau để thực hiện điều này, bao gồm:
     - **Tiến hóa (Evolutionary algorithms)**: Dựa trên nguyên tắc chọn lọc tự nhiên, các kiến trúc tốt sẽ được "duy trì" và cải tiến dần qua các thế hệ.
     - **Tối ưu hóa gradient**: Sử dụng phương pháp tối ưu hóa gradient để tìm kiếm kiến trúc tốt nhất.
     - **Học tăng cường (Reinforcement Learning)**: Sử dụng một agent để khám phá không gian tìm kiếm và tối ưu hóa kiến trúc dựa trên phần thưởng nhận được (ví dụ, độ chính xác của mô hình).
     - **Random search**: Tìm kiếm ngẫu nhiên trong không gian kiến trúc, đôi khi cũng có thể đạt kết quả tốt khi không gian tìm kiếm được thiết kế cẩn thận.
   
3. **Hàm mục tiêu (Objective Function)**:
   - Đây là tiêu chuẩn để đánh giá chất lượng của một kiến trúc mạng nơ-ron. Thông thường, hàm mục tiêu bao gồm độ chính xác của mô hình trên tập kiểm tra hoặc tập validation, nhưng cũng có thể bao gồm các yếu tố khác như thời gian huấn luyện, số phép tính toán, kích thước mô hình, và năng lượng tiêu thụ.

### Các Phương Pháp NAS Phổ Biến

#### 1. **Reinforcement Learning (Học Tăng Cường)**

Một trong những phương pháp phổ biến nhất để thực hiện NAS là sử dụng **học tăng cường (Reinforcement Learning)**. Trong phương pháp này, một agent sẽ thực hiện các hành động để khám phá không gian kiến trúc mạng. Các hành động có thể bao gồm việc chọn loại lớp, số lượng lớp, kết nối giữa các lớp, v.v. Sau khi agent tạo ra một kiến trúc nhất định, nó sẽ được huấn luyện và đánh giá trên một tập dữ liệu. Dựa trên hiệu suất của kiến trúc đó, agent sẽ nhận được một phần thưởng và cập nhật chính sách của mình để cải thiện các lựa chọn trong các lần tìm kiếm tiếp theo.

Ví dụ tiêu biểu là **NASNet** (Neural Architecture Search with Reinforcement Learning), trong đó một controller RNN được sử dụng để sinh ra các kiến trúc mạng, và phần thưởng dựa vào độ chính xác của các kiến trúc này.

#### 2. **Evolutionary Algorithm (Thuật Toán Tiến Hóa)**

Trong phương pháp này, NAS dựa vào các nguyên tắc tiến hóa để tìm kiếm kiến trúc tốt nhất. Ban đầu, một tập hợp các kiến trúc ngẫu nhiên được tạo ra (gọi là **quần thể**). Sau đó, các kiến trúc tốt nhất sẽ được chọn lọc và kết hợp với nhau (cross-over) hoặc đột biến (mutation) để tạo ra thế hệ mới. Quá trình này tiếp tục qua nhiều thế hệ cho đến khi tìm được kiến trúc tốt nhất.

Các phương pháp NAS dựa trên tiến hóa, như **AmoebaNet**, đã chứng minh khả năng tìm ra các kiến trúc mạnh mẽ với hiệu suất cao.

#### 3. **Gradient-based NAS (Tối Ưu Hóa Gradient)**

Một phương pháp gần đây là sử dụng các kỹ thuật **tối ưu hóa gradient** để tìm kiếm kiến trúc. Thay vì huấn luyện từng kiến trúc riêng lẻ và đánh giá chúng, phương pháp này sử dụng một **siêu mạng (supernet)**, nơi mà tất cả các kiến trúc mạng nơ-ron có thể được biểu diễn đồng thời. Sau đó, quá trình tìm kiếm kiến trúc sẽ sử dụng gradient để tối ưu hóa trực tiếp các kiến trúc trong siêu mạng.

Một ví dụ nổi bật của phương pháp này là **DARTS (Differentiable Architecture Search)**, trong đó các tham số kiến trúc được tối ưu hóa liên tục thông qua gradient descent.

### NAS Trong MobileNetV3

Trong **MobileNetV3**, nhóm nghiên cứu đã sử dụng NAS để tối ưu hóa kiến trúc mạng cho các tác vụ thị giác trên các thiết bị di động. MobileNetV3 là một trong những mạng nơ-ron đầu tiên kết hợp thành công NAS với các cải tiến kỹ thuật khác như **Squeeze-and-Excitation (SE) blocks** và **Hard-Swish activation**.

- **NAS trong MobileNetV3**: NAS được sử dụng để tìm ra các kiến trúc con tốt nhất cho từng tầng của mạng. Thay vì thiết kế thủ công từng thành phần, NAS tự động tìm ra các khối tích chập tối ưu, giúp giảm thiểu số phép tính toán mà vẫn tăng độ chính xác của mô hình.

- **Tối ưu hóa cho thiết bị di động**: NAS không chỉ tập trung vào độ chính xác mà còn tối ưu hóa các yếu tố khác như số phép tính FLOPs, bộ nhớ và năng lượng tiêu thụ, phù hợp với các thiết bị di động.

### Ưu Điểm Của NAS

1. **Tự động hóa quá trình thiết kế kiến trúc**: NAS giúp giảm thiểu sự can thiệp thủ công trong việc thiết kế mạng nơ-ron, tiết kiệm thời gian và công sức cho các nhà nghiên cứu.

2. **Tìm kiếm kiến trúc tối ưu**: NAS cho phép khám phá không gian kiến trúc rộng lớn và tìm ra các mô hình có độ chính xác và hiệu suất cao mà thiết kế thủ công khó có thể đạt được.

3. **Tối ưu hóa đa mục tiêu**: Ngoài việc tối ưu hóa độ chính xác, NAS có thể được thiết kế để tối ưu hóa đồng thời nhiều yếu tố khác như tốc độ, bộ nhớ, và năng lượng tiêu thụ.

### Nhược Điểm Của NAS

1. **Tốn tài nguyên tính toán**: Quá trình tìm kiếm kiến trúc trong NAS thường yêu cầu rất nhiều tài nguyên tính toán, vì mỗi kiến trúc được tìm ra cần phải được huấn luyện và đánh giá. Điều này có thể trở thành rào cản đối với các nhóm nghiên cứu có nguồn lực hạn chế.

2. **Cần tối ưu hóa không gian tìm kiếm**: Hiệu quả của NAS phụ thuộc rất nhiều vào cách không gian tìm kiếm được thiết kế. Một không gian tìm kiếm quá rộng có thể dẫn đến việc tìm kiếm tốn thời gian, trong khi một không gian tìm kiếm quá hẹp có thể bỏ lỡ các kiến trúc tốt.

3. **Khó triển khai**: NAS yêu cầu các kỹ thuật phức tạp và không phải lúc nào cũng dễ triển khai trong các hệ thống thực tế.

### Tóm Tắt

- **Neural Architecture Search (NAS)** là một phương pháp tự động hóa quá trình tìm kiếm kiến trúc mạng nơ-ron tối ưu, giúp giảm thiểu sự can thiệp của con người và tìm ra các mô hình hiệu quả.
- NAS sử dụng các kỹ thuật như **học tăng cường**, **thuật toán tiến hóa**, và **tối ưu hóa gradient** để khám phá không gian tìm kiếm.
- NAS đã được sử dụng thành công trong các mô hình như **MobileNetV3**, giúp tối ưu hóa các kiến trúc mạng nơ-ron cho các thiết bị di động về cả độ chính xác và hiệu suất tính toán.
- **Ưu điểm** của NAS là khả năng tự động hóa và tìm kiếm kiến trúc tối ưu, trong khi **nhược điểm** là tốn tài nguyên tính toán và độ phức tạp trong triển khai.

NAS mở ra một hướng đi mới cho việc thiết kế kiến trúc mạng nơ-ron, đặc biệt trong bối cảnh các yêu cầu về tối ưu hóa tài nguyên ngày càng trở nên quan trọng trong các ứng dụng thực tế như trên thiết bị di động.

### Squeeze-and-Excitation (SE) Block: Giới Thiệu

**Squeeze-and-Excitation (SE) Block** là một kỹ thuật được giới thiệu trong bài báo **"Squeeze-and-Excitation Networks"** (2018) của nhóm nghiên cứu từ Đại học Quốc gia Singapore. Squeeze-and-Excitation là một cơ chế **chú ý (attention)**, giúp mô hình học cách tập trung vào các kênh thông tin quan trọng hơn trong quá trình trích xuất đặc trưng (features). SE block được tích hợp vào nhiều kiến trúc mạng nơ-ron, bao gồm **MobileNetV3**, để cải thiện khả năng trích xuất đặc trưng mà không làm tăng đáng kể số lượng tính toán.

### Động Lực Phát Triển SE Block

Các mạng nơ-ron tích chập (CNN) truyền thống chủ yếu tập trung vào việc trích xuất thông tin không gian từ các đầu vào, nhưng chúng không có cơ chế rõ ràng để học cách tập trung vào các kênh thông tin quan trọng hơn (ví dụ, kênh chứa nhiều thông tin về đối tượng trong ảnh). Trong các CNN thông thường, mọi kênh đầu ra từ một tầng tích chập đều được xử lý một cách đồng đều, bất kể tầm quan trọng của chúng đối với nhiệm vụ hiện tại.

**SE block** ra đời để giải quyết vấn đề này, với mục tiêu:

- **Nhấn mạnh các kênh quan trọng**: SE block học cách "chú ý" đến các kênh đặc trưng quan trọng hơn trong quá trình trích xuất đặc trưng, từ đó cải thiện hiệu suất của mô hình.
  
- **Giảm bớt kênh không quan trọng**: Đồng thời giảm ảnh hưởng của các kênh ít quan trọng, giúp mô hình tập trung tốt hơn vào các thông tin cần thiết.

### Cấu Trúc Của SE Block

SE block được chèn vào giữa các tầng tích chập của mạng nơ-ron để thực hiện hai giai đoạn chính:

1. **Squeeze**: Nén không gian đặc trưng để lấy ra thông tin toàn cục (global information).
   
2. **Excitation**: Học cách khuếch đại (excite) các kênh quan trọng và giảm trọng số của các kênh không quan trọng.

#### 1. **Squeeze (Nén)**

Trong bước này, SE block nén không gian đặc trưng của đầu vào về một vector đặc trưng đại diện cho mỗi kênh. Điều này được thực hiện bằng cách tính **Global Average Pooling (GAP)** trên mỗi kênh đầu ra của lớp tích chập. GAP lấy trung bình các giá trị trong không gian của mỗi kênh, từ đó nén một đặc trưng không gian kích thước \( H \times W \) thành một giá trị đơn lẻ.

Cụ thể, nếu đầu vào có \( C \) kênh, với kích thước không gian \( H \times W \), đầu ra của bước "Squeeze" sẽ là một vector có kích thước \( C \), trong đó mỗi phần tử đại diện cho một kênh.

Công thức toán học của GAP cho kênh \( c \) là:
\[
z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_c(i, j)
\]
Trong đó \( X_c(i, j) \) là giá trị tại vị trí \( (i, j) \) của kênh \( c \).

#### 2. **Excitation (Khuếch Đại)**

Sau khi nén không gian đặc trưng thành một vector, SE block thực hiện bước "Excitation", nơi mà vector này được dùng để tính trọng số cho mỗi kênh thông qua một mô hình mạng nơ-ron nhỏ. Mục tiêu của bước này là học cách điều chỉnh trọng số của từng kênh dựa trên tầm quan trọng của chúng.

Bước này bao gồm hai tầng fully-connected (FC) và các hàm kích hoạt:

- **Tầng 1 (FC)**: Giảm số chiều của vector đầu ra từ \( C \) xuống \( C/r \), trong đó \( r \) là hệ số giảm chiều (thường được chọn là 4 hoặc 16). Điều này giúp giảm độ phức tạp của mô hình và tránh overfitting.
  
- **Hàm kích hoạt ReLU**: Hàm kích hoạt phi tuyến được sử dụng để thêm tính phi tuyến cho mô hình.

- **Tầng 2 (FC)**: Mở rộng kích thước của vector trở lại \( C \).

- **Hàm kích hoạt Sigmoid**: Được sử dụng để chuẩn hóa giá trị của mỗi phần tử trong vector về khoảng \( [0, 1] \), tạo ra một "mặt nạ" (mask) cho các kênh, qua đó điều chỉnh trọng số của các kênh.

Công thức toán học của bước "Excitation" là:
\[
s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))
\]
Trong đó:
- \( z \) là vector sau bước "Squeeze".
- \( W_1 \) và \( W_2 \) là trọng số của các tầng fully-connected.
- \( \sigma \) là hàm sigmoid.
- \( s \) là vector trọng số của các kênh.

#### 3. **Recalibration (Điều Chỉnh Lại)**

Cuối cùng, vector trọng số \( s \) được nhân theo từng kênh với đầu vào ban đầu (broadcasting trên không gian \( H \times W \) của mỗi kênh). Điều này giúp "nhấn mạnh" các kênh quan trọng và "giảm nhẹ" các kênh ít quan trọng hơn.

Công thức của bước này là:
\[
\tilde{X}_c = s_c \cdot X_c
\]
Trong đó \( X_c \) là đầu vào của kênh \( c \), và \( s_c \) là trọng số đã học sau bước "Excitation".

### Toán Học Của SE Block

SE block có ba bước chính:

1. **Squeeze**: Tính Global Average Pooling cho mỗi kênh để nhận được vector đại diện \( z \).
   
2. **Excitation**: Tính trọng số cho mỗi kênh thông qua một mô hình mạng nơ-ron nhỏ với hai tầng fully-connected và các hàm kích hoạt.

3. **Recalibration**: Điều chỉnh lại đầu vào ban đầu bằng cách nhân với trọng số đã học được ở bước "Excitation".

### Tích Hợp SE Block Vào Kiến Trúc CNN

SE block có thể dễ dàng được tích hợp vào các kiến trúc CNN hiện có như **ResNet**, **Inception**, **MobileNet**, hoặc các mạng nơ-ron tích chập khác. Thông thường, SE block được chèn vào giữa các lớp tích chập để điều chỉnh trọng số của các kênh đầu ra sau khi tích chập.

Ví dụ, trong **MobileNetV3**, SE block được tích hợp sau các khối Inverted Residual Block để điều chỉnh trọng số của các kênh và cải thiện khả năng trích xuất đặc trưng.

### Hiệu Quả Của SE Block

SE block đã chứng minh hiệu quả trong nhiều kiến trúc mạng nơ-ron khác nhau. Các ưu điểm chính bao gồm:

1. **Cải thiện độ chính xác**: SE block giúp mạng nơ-ron đạt được độ chính xác cao hơn bằng cách học cách tập trung vào các kênh quan trọng hơn.
   
2. **Hiệu quả về mặt tính toán**: SE block chỉ thêm một lượng nhỏ phép tính toán và tham số, nhưng lại mang lại cải thiện đáng kể về hiệu suất mô hình.

3. **Khả năng mở rộng**: SE block có thể dễ dàng tích hợp vào hầu hết các kiến trúc CNN hiện có mà không cần phải thay đổi cấu trúc cơ bản của chúng.

### Ưu Điểm Của SE Block

1. **Tăng độ chính xác**: SE block giúp mạng nơ-ron tập trung vào các kênh đặc trưng quan trọng hơn, từ đó cải thiện khả năng trích xuất đặc trưng và tăng độ chính xác trong các tác vụ nhận dạng hình ảnh.

2. **Chi phí tính toán thấp**: SE block chỉ thêm một lượng nhỏ phép tính toán và tham số, nhưng mang lại hiệu quả lớn về mặt cải thiện hiệu suất.

3. **Dễ tích hợp**: SE block có thể được chèn vào bất kỳ kiến trúc CNN nào mà không cần thay đổi nhiều về cấu trúc. Điều này làm cho SE block trở thành một phương pháp dễ áp dụng và có tính linh hoạt cao.

### Nhược Điểm Của SE Block

1. **Tăng độ phức tạp**: Mặc dù SE block thêm rất ít phép tính toán, nhưng vẫn tạo ra một số độ phức tạp cho mô hình, đặc biệt là đối với các mô hình rất nhẹ như **MobileNet**.

2. **Chi phí bộ nhớ**: SE block thêm các tầng fully-connected, điều này có thể làm tăng một chút yêu cầu về bộ nhớ, mặc dù không đáng kể.

### SE Block Trong MobileNetV3

Trong **MobileNetV3**, SE block được tích hợp vào các **Inverted Residual Blocks** để cải thiện khả năng trích xuất đặc trưng mà không làm tăng đáng kể số lượng phép tính toán. Cụ thể, SE block được sử dụng trong các khối tích chập có số lượng kênh lớn, nơi mà việc điều chỉnh trọng số của các kênh có thể mang lại hiệu quả cao nhất.

Kết hợp **SE block** với **Hard-Swish activation** và **Neural Architecture Search (NAS)** giúp MobileNetV3 đạt hiệu suất cao hơn so với các phiên bản trước đó như MobileNetV1 và V2.

### Tóm Tắt

- **Squeeze-and-Excitation (SE) Block** là một kỹ thuật chú ý giúp mạng nơ-ron tập trung vào các kênh quan trọng hơn trong quá trình trích xuất đặc trưng.
  
- SE block bao gồm ba bước chính: **Squeeze** (nén thông tin không gian thành một vector), **Excitation** (học cách điều chỉnh trọng số của các kênh), và **Recalibration** (điều chỉnh lại đầu vào ban đầu bằng trọng số đã học).

- **Ưu điểm** của SE block là cải thiện độ chính xác của mạng nơ-ron mà không làm tăng quá nhiều tính toán và dễ dàng tích hợp vào các kiến trúc CNN hiện có.

- **Nhược điểm** là thêm một chút độ phức tạp và chi phí bộ nhớ, nhưng nhìn chung, SE block vẫn là một cải tiến hiệu quả và được ứng dụng rộng rãi trong các mô hình hiện đại như **MobileNetV3**.

SE block đã trở thành một kỹ thuật phổ biến trong thiết kế mạng nơ-ron hiện đại, giúp tăng cường hiệu suất một cách đáng kể trên các tác vụ như nhận dạng hình ảnh và phân loại.

### EfficientNet: Giới Thiệu

**EfficientNet** là một dòng kiến trúc mạng nơ-ron tích chập (CNN) được giới thiệu trong bài báo **"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"** vào năm 2019 bởi nhóm nghiên cứu của **Google AI**. Mục tiêu chính của EfficientNet là tìm ra phương pháp hiệu quả để **tăng kích thước mô hình (model scaling)**, giúp đạt hiệu suất cao trên các tác vụ thị giác máy tính như phân loại hình ảnh, mà vẫn giữ được sự cân bằng giữa độ chính xác và tài nguyên tính toán (FLOPs).

### Động Lực Phát Triển EfficientNet

Trong quá trình phát triển các mạng nơ-ron tích chập, một vấn đề phổ biến là làm thế nào để mở rộng mô hình (scaling) nhằm đạt được hiệu suất cao hơn mà không làm tăng quá nhiều độ phức tạp tính toán. Có ba cách phổ biến để mở rộng mô hình:

1. **Tăng chiều sâu (Depth Scaling)**: Tăng số lượng lớp tích chập (convolutional layers).
   
2. **Tăng chiều rộng (Width Scaling)**: Tăng số lượng kênh trong mỗi lớp tích chập.
   
3. **Tăng độ phân giải đầu vào (Resolution Scaling)**: Tăng kích thước không gian của đầu vào (ví dụ, từ 224x224 lên 512x512).

Tuy nhiên, việc mở rộng mô hình chỉ theo một chiều (depth, width, hoặc resolution) thường dẫn đến mô hình không cân đối và có thể gây ra vấn đề quá khớp (overfitting) hoặc dưới khớp (underfitting). **EfficientNet** ra đời để giải quyết vấn đề này bằng cách đề xuất một phương pháp mở rộng mô hình **cân bằng** cả ba yếu tố này, được gọi là **compound scaling**.

### Mở Rộng Mô Hình (Scaling) Trong EfficientNet

Một trong những đóng góp lớn nhất của EfficientNet là phương pháp **compound scaling**. Thay vì chỉ tăng một yếu tố (depth, width, hoặc resolution), EfficientNet tăng đồng thời cả ba yếu tố theo một tỷ lệ đã được tối ưu hóa.

#### Công Thức Compound Scaling

Compound scaling được điều chỉnh theo một công thức đơn giản:

\[
\text{depth:} \, d = \alpha^\phi, \quad \text{width:} \, w = \beta^\phi, \quad \text{resolution:} \, r = \gamma^\phi
\]

Trong đó:

- \( \phi \) là hệ số mở rộng (scaling coefficient), điều chỉnh mức độ mở rộng của mô hình.
- \( \alpha \), \( \beta \), và \( \gamma \) là các hằng số được xác định trước dựa trên việc tối ưu hóa để duy trì sự cân bằng giữa chiều sâu, chiều rộng và độ phân giải.
- Các giá trị \( \alpha \), \( \beta \), \( \gamma \) đều được thiết lập sao cho \( \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \), nghĩa là khi \( \phi \) tăng lên, số phép tính toán (FLOPs) sẽ tăng lên khoảng gấp đôi.

### EfficientNet-B0: Kiến Trúc Cơ Bản

EfficientNet bắt đầu với một kiến trúc cơ bản gọi là **EfficientNet-B0**, được thiết kế thông qua **Neural Architecture Search (NAS)**. EfficientNet-B0 là một mô hình nhẹ nhưng hiệu quả, được sử dụng làm cơ sở để mở rộng cho các phiên bản tiếp theo (B1, B2, ..., B7).

Kiến trúc của EfficientNet-B0 được xây dựng dựa trên **MobileNetV2** và các khối **MBConv** (Mobile Inverted Bottleneck Convolutional Blocks). Ngoài ra, **Squeeze-and-Excitation (SE) blocks** cũng được tích hợp vào các khối MBConv để cải thiện khả năng trích xuất đặc trưng.

#### MBConv Block

MBConv block là một biến thể của **Inverted Residual Block** trong MobileNetV2. Nó bao gồm các thành phần chính sau:

- **Depthwise separable convolution**: Một phép tích chập được chia thành hai bước, giúp giảm số lượng phép tính toán.
- **Squeeze-and-Excitation (SE) block**: Giúp mô hình học cách tập trung vào các kênh thông tin quan trọng hơn.
- **Skip connection**: Kết nối tắt giữa đầu vào và đầu ra của khối nếu kích thước không gian và kênh bằng nhau, giúp tránh mất mát thông tin khi qua nhiều lớp.

### EfficientNet-B0: Cấu Trúc Chi Tiết

EfficientNet-B0 có cấu trúc gồm nhiều khối MBConv xếp chồng lên nhau. Dưới đây là bảng mô tả chi tiết các khối tích chập trong mô hình EfficientNet-B0:

| Stage | Operator             | Resolution | #Channels | #Layers | SE | Stride |
|-------|----------------------|------------|-----------|---------|----|--------|
| 1     | Conv3x3               | 224×224    | 32        | 1       |    | 2      |
| 2     | MBConv1, k3x3         | 112×112    | 16        | 1       |    | 1      |
| 3     | MBConv6, k3x3         | 112×112    | 24        | 2       |    | 2      |
| 4     | MBConv6, k5x5, SE     | 56×56      | 40        | 2       | ✓  | 2      |
| 5     | MBConv6, k3x3, SE     | 28×28      | 80        | 3       | ✓  | 2      |
| 6     | MBConv6, k5x5, SE     | 14×14      | 112       | 3       | ✓  | 1      |
| 7     | MBConv6, k5x5, SE     | 14×14      | 192       | 4       | ✓  | 2      |
| 8     | MBConv6, k3x3, SE     | 7×7        | 320       | 1       | ✓  | 1      |
| 9     | Conv1x1 + Pooling + FC| 7×7        | 1280      | 1       |    |        |

- **MBConv1, k3x3**: MBConv với hệ số mở rộng (expand ratio) là 1 và kernel size là 3x3.
- **MBConv6, k3x3 và k5x5**: MBConv với hệ số mở rộng là 6, kernel size là 3x3 hoặc 5x5.
- **SE**: Các khối có tích hợp Squeeze-and-Excitation block để điều chỉnh trọng số của các kênh.

### Các Phiên Bản EfficientNet (B0 đến B7)

EfficientNet có nhiều phiên bản từ **B0** đến **B7**, trong đó **B0** là phiên bản cơ bản, và các phiên bản khác được tạo ra bằng cách mở rộng mô hình theo phương pháp **compound scaling**.

| Model       | Input Resolution | #Params (Million) | FLOPs (Billion) | Top-1 Accuracy (%) |
|-------------|------------------|-------------------|-----------------|--------------------|
| EfficientNet-B0 | 224×224          | 5.3               | 0.39            | 77.3               |
| EfficientNet-B1 | 240×240          | 7.8               | 0.70            | 79.2               |
| EfficientNet-B2 | 260×260          | 9.2               | 1.0             | 80.3               |
| EfficientNet-B3 | 300×300          | 12                | 1.8             | 81.7               |
| EfficientNet-B4 | 380×380          | 19                | 4.2             | 82.9               |
| EfficientNet-B5 | 456×456          | 30                | 9.9             | 83.6               |
| EfficientNet-B6 | 528×528          | 43                | 19.0            | 84.0               |
| EfficientNet-B7 | 600×600          | 66                | 37.0            | 84.4               |

- **B0**: Là phiên bản cơ bản với số tầng và số kênh ít nhất.
- **B1 đến B7**: Các phiên bản này được mở rộng theo phương pháp compound scaling, với kích thước mô hình, số phép tính toán, và độ phân giải đầu vào tăng dần.

### Ưu Điểm Của EfficientNet

1. **Hiệu suất cao**: EfficientNet đạt được hiệu suất rất cao trên các tác vụ thị giác máy tính, đặc biệt là trên tập dữ liệu ImageNet. Phiên bản **EfficientNet-B7** đạt **84.4% độ chính xác** trên ImageNet, vượt qua nhiều mô hình lớn khác như ResNet và Inception.

2. **Hiệu quả tính toán**: EfficientNet sử dụng ít tài nguyên tính toán hơn so với nhiều mô hình khác nhưng vẫn đạt được hoặc vượt qua hiệu suất của chúng. Điều này rất quan trọng trong các ứng dụng yêu cầu tính toán giới hạn như trên thiết bị di động.

3. **Mở rộng cân bằng**: Compound scaling giúp mô hình mở rộng theo cách cân bằng, tránh được các vấn đề của việc mở rộng không cân đối (chỉ tăng depth, width, hoặc resolution).

4. **Được tối ưu hóa bởi NAS**: EfficientNet-B0 được thiết kế tự động thông qua **Neural Architecture Search (NAS)**, giúp tối ưu hóa kiến trúc mạng ngay từ đầu.

### Nhược Điểm Của EfficientNet

1. **Quá trình huấn luyện tốn kém**: Mặc dù EfficientNet tiết kiệm tài nguyên khi triển khai, quá trình **Neural Architecture Search (NAS)** để thiết kế EfficientNet-B0 yêu cầu rất nhiều tài nguyên tính toán.

2. **Độ phức tạp khi mở rộng**: Mặc dù compound scaling là một phương pháp mạnh mẽ, nhưng việc điều chỉnh các thông số \( \alpha \), \( \beta \), và \( \gamma \) cần phải được thực hiện cẩn thận để đảm bảo sự cân bằng.

### Tóm Tắt

- **EfficientNet** là một kiến trúc CNN tiên tiến sử dụng phương pháp **compound scaling** để mở rộng mô hình một cách cân bằng giữa chiều sâu, chiều rộng, và độ phân giải.
  
- EfficientNet được thiết kế tự động thông qua **Neural Architecture Search (NAS)** và sử dụng **MBConv blocks** kết hợp với **Squeeze-and-Excitation (SE) blocks** để tăng hiệu suất trích xuất đặc trưng.

- EfficientNet có các phiên bản từ **B0 đến B7**, với kích thước mô hình và độ chính xác tăng dần.

- **Ưu điểm** của EfficientNet là khả năng đạt hiệu suất cao với tài nguyên tính toán thấp, nhưng **nhược điểm** là quá trình thiết kế ban đầu thông qua NAS đòi hỏi nhiều tài nguyên.

EfficientNet đã trở thành một trong những kiến trúc CNN phổ biến nhất nhờ sự kết hợp giữa hiệu suất và hiệu quả tính toán, đặc biệt là trong các ứng dụng yêu cầu mô hình nhẹ và nhanh.

### ConvNeXt: Một Mạng Tích Chập Ấn Tượng Sau EfficientNet

Sau sự ra đời của **EfficientNet** với những cải tiến mạnh mẽ về hiệu quả tính toán và độ chính xác, có một xu hướng mới trong nghiên cứu mạng nơ-ron tích chập (CNN) là tiếp tục tinh chỉnh và cải tiến các kiến trúc CNN hiện đại để phù hợp với các yêu cầu tính toán ngày càng cao. Một trong những kiến trúc CNN nổi bật tiếp sau EfficientNet là **ConvNeXt**, được giới thiệu trong bài báo **"A ConvNet for the 2020s"** (2022) bởi nhóm nghiên cứu của **Facebook AI Research (FAIR)**.

### Động Lực Phát Triển ConvNeXt

Trong những năm gần đây, các mô hình **Transformer**, đặc biệt là **Vision Transformers (ViT)**, đã đạt được những bước tiến lớn trong lĩnh vực thị giác máy tính, khiến nhiều người nghi ngờ tính hiệu quả của CNN truyền thống. Tuy nhiên, ConvNeXt ra đời để cho thấy rằng các **mạng tích chập (ConvNets)** vẫn có thể cạnh tranh với các mô hình Transformer hiện đại nếu được thiết kế cẩn thận và áp dụng các kỹ thuật tối ưu hóa phù hợp.

Mục tiêu chính của ConvNeXt là **đơn giản hóa** và **tinh chỉnh** kiến trúc CNN để đạt hiệu suất cao hơn mà không dựa vào các kỹ thuật phức tạp hoặc thay đổi quá lớn về kiến trúc. ConvNeXt thể hiện rằng với các cải tiến nhỏ nhưng nhất quán, CNN vẫn có thể đạt được hiệu suất tốt, thậm chí cạnh tranh với các mô hình ViT.

### Kiến Trúc Của ConvNeXt

ConvNeXt được thiết kế với nhiều cải tiến xuất hiện trong các kiến trúc hiện đại như **Vision Transformers (ViT)**, nhưng vẫn giữ nguyên các đặc tính cơ bản của mạng tích chập truyền thống. Kiến trúc này lấy cảm hứng từ **ResNet**, một trong những mạng CNN phổ biến nhất, và cải tiến nó với một loạt các tinh chỉnh để tăng khả năng cạnh tranh với các mô hình Transformer.

#### Các Cải Tiến Chính Trong ConvNeXt

ConvNeXt sử dụng các cải tiến sau để đơn giản hóa và tinh chỉnh kiến trúc CNN:

1. **Thay đổi từ Conv3x3 sang Conv7x7**:
   - Thay vì sử dụng các lớp tích chập nhỏ \( 3 \times 3 \) như trong ResNet, ConvNeXt sử dụng các lớp tích chập \( 7 \times 7 \) với stride 4 để giảm độ phức tạp tính toán trong các tầng đầu tiên, đồng thời giúp mô hình học được nhiều thông tin toàn cục hơn.

2. **Thay thế BatchNorm bằng LayerNorm**:
   - ConvNeXt sử dụng **Layer Normalization (LayerNorm)** thay cho **Batch Normalization (BatchNorm)**, một kỹ thuật phổ biến trong các mô hình Transformer. Điều này giúp đơn giản hóa quá trình huấn luyện và cải thiện hiệu suất, đặc biệt là trong các bài toán thị giác máy tính.

3. **Gộp các lớp tích chập**:
   - Thay vì sử dụng nhiều lớp tích chập nhỏ, ConvNeXt gộp các lớp lại thành một lớp tích chập lớn duy nhất. Điều này giúp giảm số lượng tham số và tăng hiệu quả tính toán.

4. **Sử dụng GELU thay vì ReLU**:
   - ConvNeXt thay thế hàm kích hoạt **ReLU** bằng **GELU (Gaussian Error Linear Unit)**, một hàm kích hoạt phi tuyến đã được chứng minh là hiệu quả hơn trong các mô hình Transformer.

5. **Sử dụng các khối tích chập sâu hơn**:
   - Mỗi khối tích chập sâu hơn và được thiết kế để giữ lại nhiều thông tin hơn thông qua việc sử dụng **skip connections** và các kỹ thuật khác như **depthwise separable convolutions**.

6. **Tăng cường sử dụng DropPath**:
   - ConvNeXt áp dụng kỹ thuật **DropPath**, một biến thể của Dropout, để giảm thiểu overfitting và tăng khả năng khái quát của mô hình.

#### Cấu Trúc Chi Tiết Của ConvNeXt

ConvNeXt được xây dựng dựa trên các cải tiến từ kiến trúc **ResNet**, nhưng với các thay đổi quan trọng để tối ưu hóa các tầng tích chập. ConvNeXt bao gồm các **ConvNeXt blocks**, trong đó mỗi block được thiết kế sao cho đơn giản nhưng mạnh mẽ, giúp tăng cường khả năng học hỏi của mô hình mà không cần sử dụng nhiều tầng phức tạp.

Một ConvNeXt block có cấu trúc như sau:

1. **Conv7x7** (giảm độ phân giải với stride 4).
2. **LayerNorm** (thay cho BatchNorm).
3. **Depthwise Convolution** (giúp giảm số lượng phép tính toán).
4. **Pointwise Convolution** (để tăng số lượng kênh).
5. **GELU Activation** (thay cho ReLU).
6. **DropPath** (để tăng khả năng khái quát).
7. **Residual Connection** (giúp giữ lại thông tin qua các lớp).

### Các Phiên Bản ConvNeXt

Tương tự như các mô hình CNN khác, ConvNeXt có nhiều phiên bản với kích thước khác nhau (từ nhỏ đến lớn) để phù hợp với các yêu cầu tính toán khác nhau. Những phiên bản này bao gồm:

- **ConvNeXt-Tiny**: Phiên bản nhỏ gọn nhất, phù hợp cho các tác vụ yêu cầu tài nguyên tính toán thấp.
- **ConvNeXt-Small**: Phiên bản trung bình, cân bằng giữa độ chính xác và tài nguyên tính toán.
- **ConvNeXt-Base**: Phiên bản cơ bản với hiệu suất mạnh mẽ.
- **ConvNeXt-Large**: Phiên bản lớn hơn, phù hợp cho các tác vụ yêu cầu độ chính xác cao.
- **ConvNeXt-XL**: Phiên bản lớn nhất, tối ưu hóa cho các tác vụ cần hiệu suất cao nhất.

### Hiệu Suất Của ConvNeXt

ConvNeXt đã đạt được kết quả ấn tượng trên nhiều tập dữ liệu tiêu chuẩn như **ImageNet**. Dưới đây là kết quả của ConvNeXt so với một số mô hình hiện đại khác:

| Model           | Input Resolution | #Params (Million) | FLOPs (Billion) | Top-1 Accuracy (%) |
|-----------------|------------------|-------------------|-----------------|--------------------|
| ConvNeXt-Tiny   | 224×224          | 28.6              | 4.5             | 82.1               |
| ConvNeXt-Small  | 224×224          | 50.2              | 8.7             | 83.1               |
| ConvNeXt-Base   | 224×224          | 88.6              | 15.4            | 83.8               |
| ConvNeXt-Large  | 224×224          | 197.2             | 34.4            | 84.3               |
| ConvNeXt-XL     | 224×224          | 350.2             | 60.9            | 84.6               |

ConvNeXt đạt được hiệu suất rất cao trên tập dữ liệu **ImageNet** với độ chính xác **Top-1** vượt trên **84%**, cạnh tranh trực tiếp với các mô hình **Vision Transformers (ViT)** hiện đại.

### Ưu Điểm Của ConvNeXt

1. **Hiệu suất cao**: ConvNeXt đạt được độ chính xác rất cao trên các tác vụ thị giác máy tính, thậm chí cạnh tranh với các mô hình **Vision Transformers** hiện đại.
   
2. **Đơn giản hóa kiến trúc**: ConvNeXt được thiết kế đơn giản nhưng mạnh mẽ, giúp dễ dàng triển khai và tối ưu hóa cho các hệ thống khác nhau, bao gồm cả các thiết bị có tài nguyên hạn chế.

3. **Áp dụng các kỹ thuật hiện đại**: ConvNeXt tích hợp nhiều kỹ thuật tiên tiến như **LayerNorm**, **GELU**, và **DropPath**, giúp cải thiện hiệu suất và khả năng khái quát hóa.

4. **Khả năng mở rộng linh hoạt**: ConvNeXt có nhiều phiên bản khác nhau, từ nhỏ gọn (**Tiny**) đến rất lớn (**XL**), cho phép linh hoạt sử dụng trong nhiều ứng dụng khác nhau.

### Nhược Điểm Của ConvNeXt

1. **Kích thước mô hình lớn**: Mặc dù ConvNeXt có các phiên bản nhỏ gọn, các phiên bản lớn hơn như **ConvNeXt-Large** và **ConvNeXt-XL** có số lượng tham số rất lớn, đòi hỏi tài nguyên tính toán cao.

2. **Cạnh tranh với Transformer**: Mặc dù ConvNeXt đạt hiệu suất cao, nhưng trong một số trường hợp, mô hình **Vision Transformers (ViT)** vẫn có thể vượt trội hơn, đặc biệt là trong các tác vụ yêu cầu khả năng mô hình hóa thông tin toàn cục tốt.

### Tóm Tắt

- **ConvNeXt** là một kiến trúc CNN hiện đại được thiết kế để cạnh tranh với các mô hình **Vision Transformers**, thông qua việc đơn giản hóa và tinh chỉnh các khối tích chập truyền thống như ResNet.
  
- ConvNeXt tích hợp các kỹ thuật tiên tiến như **LayerNorm**, **GELU**, và **DropPath**, giúp cải thiện hiệu suất và khả năng khái quát hóa.

- **Ưu điểm** của ConvNeXt là hiệu suất cao và khả năng mở rộng linh hoạt, trong khi **nhược điểm** là một số phiên bản lớn đòi hỏi tài nguyên tính toán cao.

ConvNeXt thể hiện rằng các mạng tích chập (ConvNets) vẫn còn rất nhiều tiềm năng và có thể cạnh tranh với các mô hình **Vision Transformers** hiện đại, đặc biệt trong các bài toán thị giác máy tính.

### Vision Transformer (ViT): Một Mô Hình Gây Ấn Tượng Mạnh Sau ConvNeXt

Sau sự phát triển của các mạng tích chập (CNN) như **EfficientNet** và **ConvNeXt**, một kiến trúc mới đã gây chấn động trong cộng đồng học sâu, đặc biệt trong lĩnh vực **thị giác máy tính (computer vision)**. Đó là **Vision Transformer (ViT)**, một mô hình dựa trên kiến trúc **Transformer** vốn đã rất thành công trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP), nhưng lần đầu tiên được áp dụng thành công cho các tác vụ thị giác.

### Động Lực Phát Triển Vision Transformer (ViT)

Kiến trúc **Transformer**, lần đầu được giới thiệu trong bài báo **"Attention is All You Need"** (2017), đã tạo nên cuộc cách mạng trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP) với việc sử dụng cơ chế **self-attention** để học các mối quan hệ giữa các từ trong chuỗi dữ liệu. Tuy nhiên, cho đến năm 2020, **CNN** vẫn là tiêu chuẩn cho các tác vụ thị giác máy tính do khả năng khai thác mạnh mẽ các đặc trưng không gian và tính cục bộ trong ảnh.

Tuy nhiên, nhóm nghiên cứu của **Google Brain** đã đưa ra một ý tưởng đột phá: **Sử dụng Transformer cho thị giác máy tính**. Thay vì dựa vào các lớp tích chập để trích xuất đặc trưng không gian, họ đã đề xuất **Vision Transformer (ViT)**, trong đó ảnh được chia thành các **patches** nhỏ và được xử lý giống như chuỗi từ trong NLP. Điều này cho phép mô hình học các mối quan hệ toàn cục trong hình ảnh một cách tự nhiên thông qua cơ chế **self-attention**.

ViT đã chứng minh rằng mô hình Transformer có thể đạt được hoặc thậm chí vượt qua hiệu suất của các mô hình CNN tiên tiến nhất trên nhiều tác vụ thị giác, đặc biệt là khi được huấn luyện với dữ liệu lớn.

### Cấu Trúc Của Vision Transformer (ViT)

Kiến trúc của **Vision Transformer** dựa trên mô hình **Transformer encoder** tiêu chuẩn, nhưng có một số điều chỉnh để phù hợp với dữ liệu hình ảnh. Dưới đây là các thành phần chính của ViT:

#### 1. **Chia Ảnh Thành Các Patches**

Thay vì xử lý toàn bộ ảnh như một ma trận 2D hoặc áp dụng phép tích chập, **ViT** chia ảnh đầu vào thành các **patches** nhỏ (thường là các ô vuông có kích thước \(16 \times 16\) pixels). Mỗi patch này được coi như một token (tương tự như từ trong NLP).

Giả sử ảnh đầu vào có kích thước \(224 \times 224 \times 3\), việc chia ảnh thành các patch kích thước \(16 \times 16\) sẽ tạo ra \(14 \times 14 = 196\) patches, và mỗi patch có 768 giá trị (tương ứng với \(16 \times 16 \times 3\)).

#### 2. **Embedding Patches**

Mỗi patch được ánh xạ thành một vector đặc trưng bằng cách sử dụng một lớp **linear projection**. Điều này giúp chuyển đổi mỗi patch thành một không gian có số chiều cố định (thường là 768 chiều).

Công thức tính embedding cho mỗi patch là:
\[
\mathbf{z}_p = \text{Flatten}(\text{Patch}) \cdot \mathbf{W}_p
\]
Trong đó \( \mathbf{W}_p \) là ma trận trọng số của lớp embedding, và \( \mathbf{z}_p \) là vector đặc trưng của patch.

#### 3. **Thêm Thông Tin Vị Trí (Positional Encoding)**

Bởi vì Transformer không có khả năng nhận biết thông tin vị trí cục bộ như CNN, ViT thêm **positional encoding** vào từng patch để duy trì thông tin về vị trí của các patches trong ảnh gốc. Positional encoding sử dụng một vector để biểu diễn vị trí của mỗi patch trong ma trận ban đầu.

#### 4. **Thêm Token Đặc Biệt (Class Token)**

Một token đặc biệt gọi là **class token** được thêm vào chuỗi các patch embeddings. Token này sẽ được sử dụng để biểu diễn toàn bộ hình ảnh sau khi trải qua quá trình self-attention. Token này có vai trò tương tự như token [CLS] trong mô hình **BERT** của NLP.

#### 5. **Transformer Encoder**

Sau khi có được chuỗi các embeddings (bao gồm cả class token), chúng được đưa vào một chuỗi các **Transformer encoders**. Mỗi encoder bao gồm hai thành phần chính:

- **Multi-Head Self-Attention (MHSA)**: Cơ chế attention giúp mô hình học mối quan hệ giữa các patches với nhau, cho phép mô hình học các thông tin toàn cục trong ảnh.
  
- **Feed-forward Network (FFN)**: Một mạng fully-connected để chuyển đổi đầu ra từ bước attention.

Mỗi encoder cũng sử dụng **Layer Normalization (LN)** và các kết nối tắt (skip connections) để cải thiện hiệu suất và độ ổn định của mô hình.

#### 6. **Dự Đoán Cuối Cùng**

Sau khi chuỗi các patches đi qua các lớp Transformer, đầu ra của **class token** được sử dụng để làm đặc trưng đầu vào cho một lớp fully-connected cuối cùng, từ đó dự đoán nhãn của toàn bộ ảnh.

### Tóm Tắt Quy Trình Hoạt Động Của ViT

1. **Chia nhỏ ảnh** thành các patches.
2. **Embedding** các patches thành các vector đặc trưng.
3. **Thêm positional encoding** để giữ thông tin về vị trí.
4. Thêm **class token** vào chuỗi các embeddings.
5. Chuỗi này đi qua nhiều lớp **Transformer encoder**.
6. Đầu ra từ class token được sử dụng để **dự đoán nhãn**.

### Hiệu Suất Của Vision Transformer

ViT đã đạt được kết quả xuất sắc trên nhiều tập dữ liệu thị giác, đặc biệt là **ImageNet**, khi được huấn luyện trên dữ liệu lớn. Tuy nhiên, điểm mạnh của ViT thực sự tỏa sáng khi mô hình được huấn luyện với các tập dữ liệu cực kỳ lớn, chẳng hạn như **JFT-300M** (một tập dữ liệu nội bộ của Google với hơn 300 triệu ảnh).

| Model       | Input Resolution | #Params (Million) | FLOPs (Billion) | Top-1 Accuracy (%) |
|-------------|------------------|-------------------|-----------------|--------------------|
| ViT-B/16    | 224×224          | 86.6              | 17.6            | 77.9               |
| ViT-L/16    | 224×224          | 307               | 61.5            | 76.5               |
| ViT-H/14    | 224×224          | 632               | 335             | 78.6               |

Khi huấn luyện trên **ImageNet**, ViT đạt được độ chính xác rất cao, nhưng hiệu suất của nó vượt trội khi huấn luyện trên dữ liệu lớn hơn. Điều này cho thấy ViT có thể tận dụng hiệu quả sức mạnh của dữ liệu với kích thước lớn, nhưng lại dễ bị overfitting khi huấn luyện trên tập dữ liệu nhỏ hơn.

### Ưu Điểm Của Vision Transformer

1. **Học mối quan hệ toàn cục**: ViT sử dụng cơ chế **self-attention** giúp mô hình học các mối quan hệ toàn cục giữa các phần khác nhau của hình ảnh, điều mà các mô hình CNN truyền thống gặp khó khăn.
   
2. **Tận dụng dữ liệu lớn**: ViT đạt hiệu suất cao nhất khi huấn luyện trên các tập dữ liệu rất lớn. Điều này làm cho ViT trở thành sự lựa chọn lý tưởng cho các ứng dụng sử dụng big data.

3. **Sử dụng kiến trúc Transformer**: Kiến trúc Transformer đã được chứng minh là rất mạnh mẽ trong NLP, và ViT cho thấy rằng nó cũng có thể áp dụng thành công trong thị giác máy tính.

4. **Đơn giản hóa kiến trúc**: ViT không cần các lớp tích chập phức tạp hoặc tầng pooling, mà chỉ dựa vào cơ chế attention và các lớp fully-connected.

### Nhược Điểm Của Vision Transformer

1. **Cần dữ liệu lớn**: ViT hoạt động kém hiệu quả trên các tập dữ liệu nhỏ hơn, và dễ bị overfitting nếu không có lượng dữ liệu đủ lớn để huấn luyện.
   
2. **Độ phức tạp tính toán cao**: ViT yêu cầu nhiều tài nguyên tính toán hơn so với các mô hình CNN truyền thống, đặc biệt là khi sử dụng các phiên bản lớn như **ViT-L/16** hoặc **ViT-H/14**.

3. **Khó huấn luyện**: ViT yêu cầu các kỹ thuật tối ưu hóa phức tạp và cần nhiều thời gian để huấn luyện so với các kiến trúc CNN.

### Tóm Tắt

- **Vision Transformer (ViT)** là một mô hình dựa trên kiến trúc **Transformer**, được giới thiệu bởi **Google Brain** và áp dụng cho các tác vụ thị giác máy tính.
  
- ViT chia ảnh thành các patches và xử lý chúng như một chuỗi dữ liệu, học các mối quan hệ toàn cục thông qua cơ chế **self-attention**.

- **Ưu điểm** của ViT là khả năng học các đặc trưng toàn cục và hiệu suất ấn tượng khi huấn luyện trên dữ liệu lớn. Tuy nhiên, **nhược điểm** là ViT yêu cầu nhiều tài nguyên tính toán và dễ bị overfitting trên các tập dữ liệu nhỏ.

Vision Transformer (ViT) đã đánh dấu một bước ngoặt lớn trong thị giác máy tính, cho thấy rằng các mô hình Transformer có thể cạnh tranh trực tiếp với các kiến trúc CNN hiện đại và thậm chí vượt qua chúng trong nhiều tác vụ.

### **Swin Transformer**: Một Mô Hình Gây Ấn Tượng Sau Vision Transformer

Sau khi **Vision Transformer (ViT)** ra đời và tạo ra một làn sóng mạnh mẽ trong thị giác máy tính, nhiều nhà nghiên cứu đã tiếp tục cải tiến và tìm cách khắc phục những hạn chế của ViT. Một trong những mô hình nổi bật sau ViT là **Swin Transformer**, một kiến trúc gây ấn tượng mạnh nhờ khả năng mở rộng và hiệu quả cao trong nhiều tác vụ thị giác khác nhau.

**Swin Transformer** (Shifted Window Transformer) được giới thiệu bởi nhóm nghiên cứu của **Microsoft Research Asia** trong bài báo **"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"** vào năm 2021. Mô hình này không chỉ cải thiện hiệu suất mà còn mở rộng khả năng áp dụng của **Transformers** trong các tác vụ yêu cầu độ phân giải cao, như **phân đoạn ảnh**, **phát hiện đối tượng**, và **truy vết ảnh**.

### Động Lực Phát Triển Swin Transformer

Mặc dù **Vision Transformer (ViT)** đã đạt được nhiều thành tựu quan trọng, nhưng nó vẫn còn một số hạn chế:

1. **Thiếu khả năng mô hình hóa thông tin cục bộ**: ViT chia ảnh thành các patches cố định và sử dụng cơ chế **self-attention** để học các mối quan hệ toàn cục. Tuy nhiên, việc này có thể làm mất đi các thông tin cục bộ quan trọng ở mức độ pixel, đặc biệt là trong các tác vụ như phân đoạn hoặc phát hiện đối tượng.
   
2. **Không phân cấp**: ViT không có cấu trúc phân cấp như các mạng tích chập (CNN), điều này khiến nó khó có thể áp dụng hiệu quả cho các tác vụ yêu cầu độ phân giải khác nhau.

3. **Yêu cầu tài nguyên tính toán lớn**: Việc tính toán attention cho tất cả các patches cùng một lúc khiến chi phí tính toán của ViT tăng đáng kể khi độ phân giải của ảnh đầu vào tăng lên.

Để khắc phục những hạn chế này, **Swin Transformer** đã được phát triển với mục tiêu tận dụng sức mạnh của Transformer trong việc học thông tin toàn cục, trong khi vẫn giữ lại khả năng mô hình hóa các đặc trưng cục bộ và phân cấp như trong CNN.

### Cấu Trúc Của Swin Transformer

**Swin Transformer** khác biệt với Vision Transformer ở chỗ nó sử dụng một cơ chế attention phân cấp và cục bộ dựa trên **cửa sổ trượt (shifted windows)**. Kiến trúc này giúp giảm chi phí tính toán đồng thời giữ lại được thông tin cục bộ quan trọng và dễ dàng mở rộng cho các tác vụ ở độ phân giải cao.

#### 1. **Chia Ảnh Thành Các Patches**

Tương tự như ViT, Swin Transformer cũng chia ảnh đầu vào thành các **patches** nhỏ. Tuy nhiên, thay vì xử lý tất cả các patches cùng một lúc, Swin Transformer chia chúng thành các **cửa sổ không chồng lấp (non-overlapping windows)**.

#### 2. **Self-Attention Trong Cửa Sổ Cục Bộ (Local Window Self-Attention)**

Trong mỗi cửa sổ, Swin Transformer áp dụng cơ chế **self-attention** để học mối quan hệ giữa các patches **bên trong** cửa sổ đó. Điều này giúp giảm đáng kể độ phức tạp tính toán so với việc tính toàn bộ attention cho toàn bộ ảnh.

- Giả sử ảnh đầu vào có kích thước \(H \times W\), việc chia ảnh thành các cửa sổ kích thước \(M \times M\) giúp giảm độ phức tạp từ \(O((HW)^2)\) xuống còn \(O((M^2)\frac{HW}{M^2}) = O(HW)\), tiết kiệm đáng kể tài nguyên tính toán.

#### 3. **Cơ Chế Cửa Sổ Trượt (Shifted Window Mechanism)**

Một vấn đề khi chỉ áp dụng self-attention cục bộ trong từng cửa sổ là mô hình không thể nắm bắt được các mối quan hệ **giữa các cửa sổ**. Để giải quyết điều này, Swin Transformer sử dụng **cửa sổ trượt (shifted windows)**. Cụ thể, sau khi áp dụng self-attention trong các cửa sổ ban đầu, Swin Transformer trượt các cửa sổ một khoảng cố định để tạo ra một bộ cửa sổ mới và tiếp tục áp dụng attention.

Cơ chế **shifted windows** này giúp mô hình học được các mối quan hệ giữa các cửa sổ mà không làm tăng đáng kể độ phức tạp tính toán.

#### 4. **Cấu Trúc Phân Cấp (Hierarchical Structure)**

Một đặc điểm nổi bật của Swin Transformer so với ViT là nó có cấu trúc **phân cấp** tương tự như CNN. Trong quá trình xử lý, kích thước của các cửa sổ sẽ giảm dần, trong khi số lượng kênh (channel) sẽ tăng lên, tương tự như cách các tầng tích chập trong CNN hoạt động.

Điều này giúp Swin Transformer có thể học các đặc trưng ở nhiều cấp độ khác nhau, từ các đặc trưng cục bộ chi tiết đến các đặc trưng toàn cục của toàn bộ ảnh, giúp mô hình hoạt động hiệu quả hơn trong các tác vụ thị giác phức tạp.

#### 5. **Patch Merging**

Để tạo ra cấu trúc phân cấp, Swin Transformer sử dụng kỹ thuật **patch merging**, trong đó các patches liền kề được gộp lại với nhau để giảm độ phân giải không gian, đồng thời tăng số lượng kênh. Điều này tương tự như việc sử dụng **pooling layers** trong CNN để giảm kích thước không gian.

### Hiệu Suất Của Swin Transformer

Swin Transformer đã đạt được hiệu suất cực kỳ ấn tượng trên nhiều tập dữ liệu thị giác máy tính, bao gồm **ImageNet** cho phân loại ảnh, **COCO** cho nhận diện đối tượng, và **ADE20K** cho phân đoạn ảnh.

#### Kết Quả Trên Tập Dữ Liệu ImageNet

Swin Transformer đạt được kết quả rất cao trên **ImageNet**, một trong những tập dữ liệu phân loại ảnh lớn nhất và phổ biến nhất.

| Model           | Input Resolution | #Params (Million) | FLOPs (Billion) | Top-1 Accuracy (%) |
|-----------------|------------------|-------------------|-----------------|--------------------|
| Swin-Tiny       | 224×224          | 28.3              | 4.5             | 81.3               |
| Swin-Small      | 224×224          | 49.6              | 8.7             | 83.0               |
| Swin-Base       | 224×224          | 88.0              | 15.4            | 83.5               |
| Swin-Base       | 384×384          | 88.0              | 47.0            | 84.5               |
| Swin-Large      | 224×224          | 197.0             | 34.5            | 84.5               |
| Swin-Large      | 384×384          | 197.0             | 103.9           | 86.4               |

#### Kết Quả Trên Tập Dữ Liệu COCO (Phát Hiện Đối Tượng)

Với bài toán nhận diện đối tượng trên tập dữ liệu **COCO**, Swin Transformer cũng đạt được hiệu suất vượt trội.

| Model           | #Params (Million) | FLOPs (Billion) | mAP (%) |
|-----------------|-------------------|-----------------|---------|
| Swin-Tiny       | 48.0              | 264.0           | 50.1    |
| Swin-Small      | 69.1              | 359.0           | 51.8    |
| Swin-Base       | 107.0             | 496.0           | 51.9    |
| Swin-Large      | 145.0             | 838.0           | 53.5    |

#### Kết Quả Trên Tập Dữ Liệu ADE20K (Phân Đoạn Ảnh)

Swin Transformer còn thể hiện xuất sắc trong các bài toán phân đoạn ảnh, một tác vụ đòi hỏi mô hình phải hiểu rõ các đặc trưng ở cả mức cục bộ và toàn cục.

| Model           | #Params (Million) | FLOPs (Billion) | mIoU (%) |
|-----------------|-------------------|-----------------|----------|
| Swin-Tiny       | 60.0              | 945.0           | 44.5     |
| Swin-Small      | 81.0              | 1038.0          | 45.2     |
| Swin-Base       | 121.0             | 1188.0          | 46.0     |
| Swin-Large      | 234.0             | 2460.0          | 48.1     |

### Ưu Điểm Của Swin Transformer

1. **Khả năng mô hình hóa thông tin cục bộ và toàn cục**: Với cơ chế **local window attention** và **shifted windows**, Swin Transformer vừa học được các đặc trưng cục bộ giống CNN, vừa có khả năng học các đặc trưng toàn cục nhờ cơ chế self-attention.

2. **Phân cấp**: Swin Transformer có cấu trúc phân cấp tương tự như CNN, giúp mô hình dễ dàng mở rộng và áp dụng cho các tác vụ yêu cầu xử lý đa độ phân giải, như phân đoạn ảnh hoặc phát hiện đối tượng.

3. **Hiệu quả tính toán**: So với ViT, Swin Transformer giảm đáng kể chi phí tính toán nhờ vào cơ chế attention trong các cửa sổ cục bộ, thay vì tính attention trên toàn bộ ảnh.

4. **Khả năng mở rộng**: Swin Transformer hoạt động tốt ở nhiều độ phân giải khác nhau và có thể dễ dàng mở rộng cho các tác vụ yêu cầu độ phân giải cao.

### Nhược Điểm Của Swin Transformer

1. **Phức tạp hơn ViT**: Cơ chế **shifted windows** và cấu trúc phân cấp của Swin Transformer làm cho mô hình phức tạp hơn so với ViT, đòi hỏi nhiều tài nguyên hơn để huấn luyện và triển khai.

2. **Phụ thuộc vào kích thước cửa sổ**: Hiệu suất của Swin Transformer phụ thuộc nhiều vào việc lựa chọn kích thước cửa sổ. Nếu cửa sổ quá nhỏ, mô hình có thể bỏ lỡ các thông tin toàn cục. Nếu cửa sổ quá lớn, chi phí tính toán lại tăng cao.

### Tóm Tắt

- **Swin Transformer** là một cải tiến quan trọng sau Vision Transformer, kết hợp khả năng học các đặc trưng toàn cục của Transformer với khả năng mô hình hóa thông tin cục bộ và cấu trúc phân cấp của CNN.
  
- Swin Transformer đạt được hiệu suất ấn tượng trên nhiều tác vụ thị giác máy tính nhờ cơ chế **local window attention** và **shifted windows**.

- **Ưu điểm** của Swin Transformer là khả năng xử lý hiệu quả các tác vụ ở nhiều độ phân giải, trong khi **nhược điểm** là sự phức tạp trong thiết kế và phụ thuộc vào việc lựa chọn kích thước cửa sổ.

Swin Transformer đã khẳng định rằng **Transformers** không chỉ vượt trội trong xử lý ngôn ngữ tự nhiên mà còn có thể thay thế hoặc bổ sung cho các kiến trúc CNN trong nhiều tác vụ thị giác máy tính, đặc biệt là khi yêu cầu khả năng xử lý thông tin ở nhiều cấp độ khác nhau.

