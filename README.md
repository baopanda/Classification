# Bài toán phân loại văn bản
Phân loại văn bản hay còn gọi là Text Classifcation hoặc là Text Categorizer (từ bầy giờ tôi sẽ gọi tắt là TC cho tiện) là một bài toán thuộc về lĩnh vực Xử lý ngôn ngữ tự nhiên dưới dạng văn bản (text). Tuy nhiên nó gắn liền với Machine Learning bởi vì nó có từ phân loại làm cho chúng ta nhớ đến những khái niệm cơ bản mà tôi đã đề cập trong Bài 1 của blog này. Nếu các bạn là những người mới bước chân vào lĩnh vực này thì tôi xin phép được nhắc lại một chút ở đây
<img src="https://cdn-images-1.medium.com/max/700/1*ljCBykAJUnvaZcuPYwm4_A.png">

## 1.Khám Phá Dữ Liệu
Dữ liệu đầu vào là 1 file Json hơn 15mb gốm 5159 văn bản là 5159 labels tương ứng. 
<img src="https://i.imgur.com/zikhsp6.png">
### Tiền Xử Lý Dữ Liệu
* Do đây là văn bản tiếng việt nên chúng ta cần tách từ cho nó. Đây là một bước quan trọng bậc nhất trong xử lý ngôn ngữ tự nhiên, tiếng Việt không đơn giản như tiếng anh vì nó có thêm các từ ghép. Có thể tách từ theo nhiều cách khác nhau gây ra sự nhập nhằng về mặt ngữ nghĩa. Đây là một bài toán hết sức thú vị. Tuy nhiên chúng ta có một số công cụ để thực hiện việc này mà phổ biến nhất đó là VnTokenizer.
* Sau đó chúng ta cần loại bỏ stop-word. Stop-Word là những từ không có ý nghĩa phân loại đối với việc phân loại của chúng ta. Trong nltk có cho chúng ta list nhưng stop-word của nhiều ngôn ngữ nhưng rất tiếc lại không có tiếng việt nên chúng ta phải import chúng.(file stop-word tôi sẽ để trong resposity này)
<img src="https://i.imgur.com/6TiF5o4.png">

* Chúng ta phải loại bỏ một số ký tự đặc biệt và 1 số từ quá dài(do lỗi). Những từ quá dài này sẽ làm cho thư việc tách từ của chúng ta thực hiện trong thời gian rất lâu. 

<img src="https://i.imgur.com/8OKWUzz.png">

## Thuật Toán Sử Dụng:
Thuật Toán Chúng ta sử dụng là thuật toán NaiveBayes 
Lý thuyết Bayes thì có lẽ không còn quá xa lạ với chúng ta nữa rồi. Nó chính là sự liên hệ giữa các xác suất có điều kiện. Điều đó gợi ý cho chúng ta rằng chúng ta có thể tính toán một xác suất chưa biết dựa vào các xác suất có điều kiện khác. Thuật toán Naive Bayes cũng dựa trên việc tính toán các xác suất có điều kiện đó. Nghe tên thuật toán là đã thấy gì đó ngây ngô rồi. Tại sao lại là Naive nhỉ. Không phải ngẫu nhiên mà người ta đặt tên thuật toán này như thế. Tên gọi này dựa trên một giả thuyết rằng các chiều của dữ liệu X=(x_1, x_2, ...., x_n)X=(x 
1
​	
 ,x 
2
​	
 ,....,x 
n
​	
 ) là độc lập về mặt xác suất với nhau. 
 <img src="https://viblo.asia/uploads/a468626e-0831-4efb-b4be-537f5329f050.png"> Chúng ta có thể thấy rằng giả thuyết này có vẻ khá ngây thơ vì trên thực tế điều này có thể nói là không thể xảy ra tức là chúng ta rất ít khi tìm được một tập dữ liệu mà các thành phần của nó không liên quan gì đến nhau. Tuy nhiên, giả thiết ngây ngô này lại mang lại những kết quả tốt bất ngờ. Giả thiết về sự độc lập của các chiều dữ liệu này được gọi là Naive Bayes (xin phép không dịch). Cách xác định class của dữ liệu dựa trên giả thiết này có tên là Naive Bayes Classifier (NBC). Tuy nhiên dựa vào giả thuyết này mà bước training và testing trở nên vô cùng nhanh chóng và đơn giản. Chúng ta có thể sử dụng nó cho các bài toán large-scale. Trên thực tế, NBC hoạt động khá hiệu quả trong nhiều bài toán thực tế, đặc biệt là trong các bài toán phân loại văn bản

### Training 
* Ở đây có một vấn đề, các giải thuật Machine Learning chỉ làm việc được với số, nên mình sẽ convert labels và cả các data về định dạng số. 
* Tiếp theo, ta sẽ transform data thành dạng số ( dùng module mà scikit learn cung cấp ). Module mà scikit learn cung cấp cho phép chuyển đổi định dạng text thành vector, mình sẽ import CountVectorizer và transform text thành vector. Cách transform thế này: mình có một mảng các string, mình sẽ transform mảng này sao mỗi string sẽ chuyển đổi thành 1 vector có độ dài d (số từ xuất hiện ít nhất 1 lần), giá trị của thành phần thứ i trong vector chính là số lần từ đó xuất hiện trong string.
Sau đó ta sẽ sử dụng vectory vocab để lập ra từ điển các từ r ánh xạ vào file X_valid.
* Sau đó, Ta sẽ import Naive Bayes, fit rồi predict là xong. 
<img src="https://i.imgur.com/S9tEuKk.png">

* Kết quả: Có vẻ khá khả quan nhỉ
<img src="https://i.imgur.com/KNi1c69.png">

Áp dụng GridSearchCV vào mô hình naive bayes trên thì thấy Accurancy có tăng lên. Param mà mình apply Grid Search ở đây là alpha, người ta thêm nó vào cải thiện độ chính xác.

Với Grid Search, giả dụ giá trị của 2 parameter lần lượt từ 0-9. Grid Search sẽ lần lượt ghép từng giá trị của param 1 với param 2 để tính toán độ chính xác của model. Đảm bảo không bỏ sót cặp parameter nào.

Ưu điểm: Diệt nhầm còn hơn bỏ sót, nên thường được ưu tiên lựa chọn.

Nhược điểm: Tuy nhiên đối với các model cần thiết lập nhiều parameter và nhiều giá trị thì việc tunning sẽ mất rất nhiều thời gian, hàng giờ, vài giờ thậm chỉ có thể tính bằng ngày.

Kết Quả: 
<img src="https://i.imgur.com/GQFVg2D.png">

## Confusion Matrx để đánh giá mô hình
<img src="https://i.imgur.com/FGWxe1w.png">
