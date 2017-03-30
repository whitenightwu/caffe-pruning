#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/prun_cfg.hpp"

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}
  // for pruning by zhluo
template <typename Dtype>
Dtype* Blob<Dtype>::cpu_data_prun() const {
  CHECK(data_);
  return (Dtype*)data_->cpu_data();
} // for pruning by zhluo

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_gpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
	static_cast<const Dtype*>(diff_->cpu_data()),
	static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
	static_cast<const Dtype*>(diff_->gpu_data()),
	static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
	   LegacyShape(-4) == other.num() &&
	   LegacyShape(-3) == other.channels() &&
	   LegacyShape(-2) == other.height() &&
	   LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
	  static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
	  static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
	  static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
	  static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
	proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
	shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

//INSTANTIATE_CLASS(Blob);
//template class Blob<int>;
//template class Blob<unsigned int>;

  //}  // namespace caffe

// for pruning by zhluo
template <>
void Blob<float>::Update_Prun() {
  // We will perform update based on where the data is located.
  float *diff_val_cpu = (float*)diff_->cpu_data();
  float *weight_val_cpu = static_cast<float*>(data_->mutable_cpu_data());
#ifndef CPU_ONLY
  float *diff_val_gpu = (float*)diff_->gpu_data();
  float *weight_val_gpu = static_cast<float*>(data_->mutable_cpu_data());
#endif
  
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    {
      for (int i = 0; i < count_; i++)
	if (weight_val_cpu[i] == 0)
	  {
	    diff_val_cpu[i] = 0;
	  }
      caffe_axpy<float>(count_, float(-1),
			static_cast<const float*>(diff_val_cpu),
			weight_val_cpu);
      break;
    }
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
    {
#ifndef CPU_ONLY
      // perform computation on GPU
      for (int i = 0; i < count_; i++)
	if (weight_val_gpu[i] == 0)
	  {
	    diff_val_gpu[i] = 0;
	  }
      caffe_gpu_axpy<float>(count_, float(-1),
			    static_cast<const float*>(diff_val_gpu),
			    weight_val_gpu);
#else
      NO_GPU;
#endif
      break;
    }
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <>
void Blob<double>::Update_Prun() {
  // We will perform update based on where the data is located.
  double *diff_val_cpu = (double*)diff_->cpu_data();
  double *weight_val_cpu = static_cast<double*>(data_->mutable_cpu_data());
#ifndef CPU_ONLY
  double *diff_val_gpu = (double*)diff_->gpu_data();
  double *weight_val_gpu = static_cast<double*>(data_->mutable_cpu_data());
#endif
  
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    {
      for (int i = 0; i < count_; i++)
	if (weight_val_cpu[i] == 0)
	  {
	    diff_val_cpu[i] = 0;
	  }
      caffe_axpy<double>(count_, double(-1),
			 static_cast<const double*>(diff_val_cpu),
			 weight_val_cpu);
      break;
    }
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
    {
#ifndef CPU_ONLY
      // perform computation on GPU
      for (int i = 0; i < count_; i++)
	if (weight_val_gpu[i] == 0)
	  {
	    diff_val_gpu[i] = 0;
	  }
      caffe_gpu_axpy<double>(count_, double(-1),
			     static_cast<const double*>(diff_val_gpu),
			     weight_val_gpu);
#else
      NO_GPU;
#endif
      break;
    }
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <typename Dtype>
void Blob<Dtype>::CalWeightPrun(Dtype** weight, int count, bool prun, int num) const {
  if (FLAGS_prun_fc)
    {
      int prun_cnt = 0;
      Dtype* tmp_data = *weight;
      Dtype thr_weight = 0;
      vector<Dtype> sort_weight(count);
      if (prun)
	{
	  for (int i = 0; i < count; ++i)
	    sort_weight[i] = fabs(tmp_data[i]);

	  sort(sort_weight.begin(), sort_weight.end());
	  
	  if (num == 0)
	    {
	      thr_weight = sort_weight[count * FLAGS_fc_ratio_0];
	    }
	  else if (num == 1)
	    {
	      thr_weight = sort_weight[count * FLAGS_fc_ratio_1];
	    }
	  else if (num == 2)
	    {
	      thr_weight = sort_weight[count * FLAGS_fc_ratio_2];
	    }
	  else
	    {
	      LOG(FATAL) << " Error: Illegal FC ratio ";
	    }
	  
	  LOG(INFO) << "blob <FC>  threshold: " << thr_weight;
	  for (int i = 0; i < count; ++i)
	    {
	      if ((tmp_data[i] > -thr_weight) && (tmp_data[i] < thr_weight))
		{
		  tmp_data[i] = 0;
		  prun_cnt++;
		}
	    }
	  //for (int i = 0; i < count; i++)
	  //	{
	  //	  l2_weight += tmp_data[i] * tmp_data[i];
	  //	  if (max_weight < tmp_data[i])
	  //	    max_weight = tmp_data[i];
	  //	  if (min_weight > tmp_data[i])
	  //	    min_weight = tmp_data[i];
	  //	}
	  ////l2_weight = sqrt(l2_weight);
	  //LOG(INFO) << "blob <> here sqrt l2 DATA: " << l2_weight;
	  //LOG(INFO) << "blob <> max weight: " << max_weight;
	  //LOG(INFO) << "blob <> min weight: " << min_weight;
	  ////thr = fabs((max_weight + min_weight)/2);
	  ////thr = sqrt(fabs(max_weight) * fabs(min_weight))/2;
	  ////thr = pow(max_weight, 2.0) + pow(min_weight, 2.0);
	  ////beta = (max_weight + min_weight)/2;
	  //gamma = 1.0 - fabs(max_weight) - fabs(min_weight);
	  //if (count > 100000)
	  //	{
	  //	  beta = 0.045;
	  //	  gamma = 0.80;
	  //	}
	  //else
	  //	{
	  //	  beta = 0.025;
	  //	  gamma = 0.60;
	  //	}
	  //thr = pow(fabs(beta - l2_weight * alpha), gamma); 
	  //LOG(INFO) << "blob <> beta : " << beta << ", gamma: " << gamma <<", threshold: " << thr;
	  //for (int i = 0; i < count; i++)
	  //	{
	  //	  //if (tmp_data[i] > -0.05 && tmp_data[i] < 0.05)
	  //	  if (tmp_data[i] > -thr && tmp_data[i] < thr)
	  //	    {
	  //	      tmp_data[i] = 0;
	  //	      prun_cnt++;
	  //	    }
	  //	}
	  LOG(INFO) << ">total num: " << count << ", prun count: " << prun_cnt;
	}
    }
  else if (FLAGS_prun_conv)
    {
      //Dtype l2_weight = 0;
      //Dtype max_weight = 0;
      //Dtype min_weight = 0;
      //Dtype thr = 0;
      //Dtype alpha = 0.0001;
      //Dtype beta = 0.0;
      //Dtype gamma = 0.0;
  
      int prun_cnt = 0;
      Dtype* tmp_data = *weight;
      Dtype thr_weight = 0;
      vector<Dtype> sort_weight(count);
      if (prun)
	{ 
	  for (int i = 0; i < count; ++i)
	    sort_weight[i] = fabs(tmp_data[i]);

	  sort(sort_weight.begin(), sort_weight.end());
	  
	  if (num == 0)
	    {
	      thr_weight = sort_weight[count * FLAGS_conv_ratio_0];
	    }
	  else if (num == 1)
	    {
	      thr_weight = sort_weight[count * FLAGS_conv_ratio_1];
	    }
	  else if (num == 2)
	    {
	      thr_weight = sort_weight[count * FLAGS_conv_ratio_2];
	    }
	  else
	    {
	      LOG(FATAL) << " Error: Illegal FC ratio ";
	    }
	  
	  LOG(INFO) << "blob <CONV>  threshold: " << thr_weight;
	  for (int i = 0; i < count; ++i)
	    {
	      if ((tmp_data[i] > -thr_weight) && (tmp_data[i] < thr_weight))
		{
		  tmp_data[i] = 0;
		  prun_cnt++;
		}
	    }
	  //for (int i = 0; i < count; i++)
	  //	{
	  //	  l2_weight += tmp_data[i] * tmp_data[i];
	  //	  if (max_weight < tmp_data[i])
	  //	    max_weight = tmp_data[i];
	  //	  if (min_weight > tmp_data[i])
	  //	    min_weight = tmp_data[i];
	  //	}
	  ////l2_weight = sqrt(l2_weight);
	  //LOG(INFO) << "blob <> here sqrt l2 DATA: " << l2_weight;
	  //LOG(INFO) << "blob <> max weight: " << max_weight;
	  //LOG(INFO) << "blob <> min weight: " << min_weight;
	  ////thr = fabs((max_weight + min_weight)/2);
	  ////thr = sqrt(fabs(max_weight) * fabs(min_weight))/2;
	  ////thr = pow(max_weight, 2.0) + pow(min_weight, 2.0);
	  ////beta = (max_weight + min_weight)/2;
	  //gamma = 1.0 - fabs(max_weight) - fabs(min_weight);
	  //if (count > 1000)
	  //	{
	  //	  beta = 0.045;
	  //	  gamma = 0.80;
	  //	}
	  //else
	  //	{
	  //	  beta = 0.025;
	  //	  gamma = 0.60;
	  //	}
	  //thr = pow(fabs(beta - l2_weight * alpha), gamma); 
	  //LOG(INFO) << "blob <CONV> beta : " << beta << ", gamma: " << gamma <<", threshold: " << thr;
	  //for (int i = 0; i < count; i++)
	  //	{
	  //	  //if (tmp_data[i] > -0.05 && tmp_data[i] < 0.05)
	  //	  if (tmp_data[i] > -thr && tmp_data[i] < thr)
	  //	    {
	  //	      tmp_data[i] = 0;
	  //	      prun_cnt++;
	  //	    }
	  //	}
	  LOG(INFO) << ">total num: " << count << ", prun count: " << prun_cnt;
	}
    }
}

template <>
void Blob<float>::ToProtoPrun(BlobProto* proto, bool write_diff, bool prun, int num) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  //float* data_vec = cpu_data_prun();
  float* data_vec = mutable_cpu_data();

  CalWeightPrun(&data_vec, count_, prun, num);

  if (FLAGS_sparse_csc && prun)
    encode_weight(proto, &data_vec);
  else
    {
      for (int i = 0; i < count_; ++i) {
	proto->add_data(data_vec[i]);
      }
    }
  
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<double>::ToProtoPrun(BlobProto* proto, bool write_diff, bool prun, int num) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  //double* data_vec = cpu_data_prun();
  double* data_vec = mutable_cpu_data();
    
  CalWeightPrun(&data_vec, count_, prun, num);

  if (FLAGS_sparse_csc && prun)
    encode_weight(proto, &data_vec);
  else
    {
      for (int i = 0; i < count_; ++i) {
	proto->add_double_data(data_vec[i]);
      }
    }
  
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template<typename Dtype>
void Blob<Dtype>::encode_weight(BlobProto* proto, Dtype** weight) const {
  // change blob as 2D-array, row: count_/100, col:100
  int idx_num = 0;
  int ptr[101]; 
  int idx[count_];
  Dtype data[count_];
  Dtype* tmp_data = *weight;
    
  // CSC: encode weight
  for (int col = 0; col < 100; ++col)
    {
      ptr[col] = idx_num;
      
      proto->add_csc_ptr(idx_num);
      
      for (int row = 0; row < count_/100; row++)
	{
	  if (tmp_data[row*100+col] != 0)
	    {
	      data[idx_num] = tmp_data[row*100+col];
	      idx[idx_num] = row;

	      proto->add_csc_row_idx(row);
	      proto->add_csc_data(tmp_data[row*100+col]);
	      
	      idx_num++;
	    }
	}
    }
  ptr[100] = idx_num;
  proto->add_csc_ptr(idx_num);
  
  LOG(INFO) << " [Info] valid data num: " << idx_num;
    
  for (int i = 0; i < count_; ++i)
    tmp_data[i] = 0;
    
  // write CSC info into blob
  // write ptr
  for (int i = 0; i < 101; i++)
    tmp_data[i] = (Dtype)ptr[i];
  tmp_data[101] = (Dtype)100;
    
  LOG(INFO) << "[Info] weight valid data: " << tmp_data[100] << " invalid: " << tmp_data[101];
  // write index
  for(int i = 0; i < idx_num; i++)
    tmp_data[102 + i] = (Dtype)idx[i];
  tmp_data[102 + idx_num] = (Dtype)100;
  
  // write data
  for(int i = 0; i < idx_num; i++)
    tmp_data[102 + idx_num + 1 + i] = data[i];
}

template<typename Dtype>
void Blob<Dtype>::decode_weight(Dtype** weight) const {
  int idx_num = 0;
  int ptr[count_/2+1];
  int idx[count_];
  Dtype data[count_];
  Dtype* tmp_data = *weight;
  
  idx_num = tmp_data[100];
  for (int i = 0; i < count_; i++)
    data[i] = 0;
  
  for(int i = 0; i < 100; i++)
    {
      int start = ptr[i];
      int end = ptr[i+1];
      for (int j = start; j < end; j++)
	{
	  data[i + 100*idx[j]] = tmp_data[100 + 1 + idx_num + 1 + j];
	}
    }
  
  for (int i = 0; i < count_; i++)
    tmp_data[i] = data[i];
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe

