#include <climits>
#include <vector>
#include <limits>
#include <iostream>

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
  // modify for pruning, by zhluo
  if (proto.csc_ptr_size() != 0)
    {
      decode_weight(&proto, &data_vec);
    }
  else
    {
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

template <> void Blob<unsigned int>::Update_Prun() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update_Prun() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update_Prun() {
  // We will perform update based on where the data is located.
  Dtype *diff_val_cpu = (Dtype*)diff_->cpu_data();
  Dtype *weight_val_cpu = static_cast<Dtype*>(data_->mutable_cpu_data());
#ifndef CPU_ONLY
  Dtype *diff_val_gpu = (Dtype*)diff_->gpu_data();
  Dtype *weight_val_gpu = static_cast<Dtype*>(data_->mutable_cpu_data());
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
      caffe_axpy<Dtype>(count_, Dtype(-1),
			static_cast<const Dtype*>(diff_val_cpu),
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
      caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
			    static_cast<const Dtype*>(diff_val_gpu),
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
void Blob<Dtype>::Update_Quan(int* quan_data) {
  // We will perform update based on where the data is located.
  int index = 0;
  Dtype *diff_val_cpu = (Dtype*)diff_->cpu_data();
  Dtype *weight_val_cpu = static_cast<Dtype*>(data_->mutable_cpu_data());
#ifndef CPU_ONLY
  Dtype *diff_val_gpu = (Dtype*)diff_->gpu_data();
  Dtype *weight_val_gpu = static_cast<Dtype*>(data_->mutable_cpu_data());
#endif

  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    {
      int centroid_max_num = 1 << FLAGS_quan_k_max;
      Dtype diff_sum[centroid_max_num];
      for (int i = 0; i < centroid_max_num; ++i)
	diff_sum[i] = 0;
      
      index = 0;
      for (int i = 0; i < count_; ++i)
	if (weight_val_cpu[i] != 0)
	  diff_sum[quan_data[index++]-1] += diff_val_cpu[i];
      
      for (int i = 0; i < centroid_max_num; ++i)
	diff_sum[i] = diff_sum[i] * FLAGS_quan_lr;
      
      index = 0;
      for (int i = 0; i < count_; ++i)
	if (weight_val_cpu[i] != 0)
	  weight_val_cpu[i] -= diff_sum[quan_data[index++]-1];
      break;
    }
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
    {
#ifndef CPU_ONLY
      // perform computation on GPU
      int centroid_max_num = 1 << FLAGS_quan_k_max;
      Dtype diff_sum[centroid_max_num];

      for (int i = 0; i < centroid_max_num; ++i)
	diff_sum[i] = 0;
      
      index = 0;
      for (int i = 0; i < count_; ++i)
	if (weight_val_gpu[i] != 0)
	  diff_sum[quan_data[index++]-1] += diff_val_gpu[i];
      
      for (int i = 0; i < centroid_max_num; ++i)
	diff_sum[i] = diff_sum[i] * FLAGS_quan_lr;
      
      index = 0;
      for (int i = 0; i < count_; ++i)
	if (weight_val_gpu[i] != 0)
	  weight_val_gpu[i] -= diff_sum[quan_data[index++]-1];
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
int Blob<Dtype>::CalWeightPrun(Dtype** weight, int count, bool prun, int num) const {
  int prun_cnt = 0;
  Dtype* tmp_data = *weight;
  Dtype thr_weight = 0;
  vector<Dtype> sort_weight(count);

  if (FLAGS_prun_fc)
    {
      //Dtype* tmp_data = *weight;
      //Dtype thr_weight = 0;
      //vector<Dtype> sort_weight(count);
      if (prun)
	{
	  for (int i = 0; i < count; ++i)
	    sort_weight[i] = fabs(tmp_data[i]);

	  sort(sort_weight.begin(), sort_weight.end());

	  if (num == 0)
	    thr_weight = sort_weight[count * FLAGS_fc_ratio_0];
	  else if (num == 1)
	    thr_weight = sort_weight[count * FLAGS_fc_ratio_1];
	  else if (num == 2)
	    thr_weight = sort_weight[count * FLAGS_fc_ratio_2];
	  else if (num == 3)
	    thr_weight = sort_weight[count * FLAGS_fc_ratio_3];
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
      //Dtype* tmp_data = *weight;
      //Dtype thr_weight = 0;
      //vector<Dtype> sort_weight(count);
      if (prun)
	{
	  for (int i = 0; i < count; ++i)
	    sort_weight[i] = fabs(tmp_data[i]);

	  sort(sort_weight.begin(), sort_weight.end());

	  if (num == 0)
	    thr_weight = sort_weight[count * FLAGS_conv_ratio_0];
	  else if (num == 1)
	    thr_weight = sort_weight[count * FLAGS_conv_ratio_1];
	  else if (num == 2)
	    thr_weight = sort_weight[count * FLAGS_conv_ratio_2];
	  else if (num == 3)
	    thr_weight = sort_weight[count * FLAGS_conv_ratio_3];
	  else if (num == 4)
	    thr_weight = sort_weight[count * FLAGS_conv_ratio_4];
	  else if (num == 5)
	    thr_weight = sort_weight[count * FLAGS_conv_ratio_5];
	  else if (num == 6)
	    thr_weight = sort_weight[count * FLAGS_conv_ratio_6];
	  else if (num == 7)
	    thr_weight = sort_weight[count * FLAGS_conv_ratio_7];
	  else
	    {
	      //LOG(FATAL) << " Error: Illegal CONV ratio ";
	      LOG(INFO) << " [Warning] CONV layers exceed to the default value[3]," <<
		" pruning ratio use default value:<0.5>.";
	      thr_weight = sort_weight[count * 0.5];
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
	  LOG(INFO) << ">total num: " << count << ", prun count: " << prun_cnt;
	}
    }
  else
    {
      if (FLAGS_sparse_csc)
	LOG(WARNING) << " [Warning] execute sparse routine, Blob use CSC storage pattern.";
      else
	LOG(FATAL) << " [Error] please set FLAGS_prun_fc or FLAGS_prun_conv valid,"<<
	  " reference \"src/caffe/prun_cfg.cfg\".";
    }
  return (count-prun_cnt);
}

template <>
void Blob<unsigned int>::encode_weight(BlobProto* proto, unsigned int* weight, int diff_num) {
  NOT_IMPLEMENTED;
}

template <>
void Blob<int>::encode_weight(BlobProto* proto, int* weight, int diff_num) {
  NOT_IMPLEMENTED;
}

template <>
void Blob<float>::encode_weight(BlobProto* proto, float* weight, int diff_num) {
  // row: count_/FLAGS_sparse_col, col:FLAGS_sparse_col
  int idx_num = 0;
  int base_idx = 0;
  int thr_diff = 1 << diff_num;

  // CSC: encode weight(storage index diff)
  for (int col = 0; col < FLAGS_sparse_col; ++col)
    {
      proto->add_csc_ptr(idx_num);

      for (int row = 0; row < count_/FLAGS_sparse_col; row++)
	{
	  if (weight[row*FLAGS_sparse_col+col] != 0)
	    {
	      if ((row - base_idx) < thr_diff)
		{
		  proto->add_csc_row_idx(row-base_idx);
		  proto->add_csc_data(weight[row*FLAGS_sparse_col+col]);
		  base_idx = row;
		}
	      else
		{
		  do
		    {
		      proto->add_csc_row_idx(thr_diff);
		      proto->add_csc_data(0);
		      base_idx = base_idx + thr_diff;
		    }while((row-base_idx) > thr_diff);
		  proto->add_csc_row_idx(row-base_idx);
		  proto->add_csc_data(weight[row*FLAGS_sparse_col+col]);
		  base_idx = row;
		}
	      idx_num++;
	    }
	}
    }
  proto->add_csc_ptr(idx_num);

  LOG(INFO) << " [Info] CSC store float valid data num: " << idx_num;
}

template <>
void Blob<double>::encode_weight(BlobProto* proto, double* weight, int diff_num) {
  // row: count_/FLAGS_sparse_col, col:FLAGS_sparse_col
  int idx_num = 0;
  int base_idx = 0;
  int thr_diff = 1 << diff_num;

  // CSC: encode weight
  for (int col = 0; col < FLAGS_sparse_col; ++col)
    {
      proto->add_csc_ptr(idx_num);

      for (int row = 0; row < count_/FLAGS_sparse_col; row++)
	{
	  if (weight[row*FLAGS_sparse_col+col] != 0)
	    {
	      if ((row - base_idx) < thr_diff)
		{
		  proto->add_csc_row_idx(row-base_idx);
		  proto->add_double_csc_data(weight[row*FLAGS_sparse_col+col]);
		  base_idx = row;
		}
	      else
		{
		  do
		    {
		      proto->add_csc_row_idx(thr_diff);
		      proto->add_double_csc_data(0);
		      base_idx = base_idx + thr_diff;
		    }while((row-base_idx) > thr_diff);
		  proto->add_csc_row_idx(row-base_idx);
		  proto->add_double_csc_data(weight[row*FLAGS_sparse_col+col]);
		  base_idx = row;
		}
	      idx_num++;
	    }
	}
    }
  proto->add_csc_ptr(idx_num);

  LOG(INFO) << " [Info] CSC store double valid data num: " << idx_num;
}

template <>
void Blob<float>::quan_retrain(BlobProto* proto) {
  bool flag = true;
  int cen_fill_num = 0;
  int cen_num = 1 << FLAGS_quan_k_max;
  float centroid[cen_num];
  
  for (int i = 0; i < cen_num; ++i)
    centroid[i] = 0;
  
  for (int i = 0; i < proto->csc_data_size(); ++i)
    {
      flag = true;
      if (proto->csc_data(i) == 0)
	{
	  proto->add_csc_quan_data(0);
	}
      else
	{
	  for (int j = 0; j < cen_num; ++j)
	    {
	      if (centroid[j] == proto->csc_data(i))
		{
		  flag = false;
		  proto->add_csc_quan_data(j+1);
		  continue;
		}
	    }
	  if (flag)
	    {
	      centroid[cen_fill_num++] = proto->csc_data(i);
	      proto->add_csc_quan_data(cen_fill_num);
	    }
	}
    }
  cen_fill_num--;
  
  for (int i = 0; i < cen_fill_num; i++)
    proto->add_quan_data(centroid[i]);
  
  if (cen_fill_num > cen_num)
    LOG(FATAL) << " [Error] Index exceed bonudary( " << cen_fill_num << " vs " << cen_num << " ).";
  if (proto->csc_quan_data_size() != proto->csc_data_size())
    LOG(FATAL) << " [Error] Size not equal( " << proto->csc_quan_data_size() <<
      " vs " << proto->csc_data_size() << " ).";
  		  
  LOG(INFO) << " ^_^ here" << " szie: " << proto->csc_quan_data_size();
  proto->clear_csc_data();
}

template <>
void Blob<double>::quan_retrain(BlobProto* proto) {
  bool flag = true;
  int cen_fill_num = 0;
  int cen_num = 1 << FLAGS_quan_k_max;
  double centroid[cen_num];
  
  for (int i = 0; i < cen_num; ++i)
    centroid[i] = 0.f;
  
  for (int i = 0; i < proto->double_csc_data_size(); ++i)
    {
      flag = true;
      if (proto->double_csc_data(i) == 0)
	{
	  proto->add_csc_quan_data(0);
	}
      else
	{
	  for (int j = 0; j < cen_num; ++j)
	    {
	      if (centroid[j] == proto->double_csc_data(i))
		{
		  flag = false;
		  proto->add_csc_quan_data(j+1);
		  continue;
		}
	    }
	  if (flag)
	    {
	      centroid[cen_fill_num++] = proto->double_csc_data(i);
	      proto->add_csc_quan_data(cen_fill_num);
	    }
	}
    }
  cen_fill_num--;
  
  for (int i = 0; i < cen_fill_num; i++)
    proto->add_quan_data(centroid[i]);
  
  if (cen_fill_num > cen_num)
    LOG(FATAL) << " [Error] Index exceed bonudary( " << cen_fill_num << " vs " << cen_num << " ).";
  if (proto->csc_quan_data_size() != proto->double_csc_data_size())
    LOG(FATAL) << " [Error] Size not equal( " << proto->csc_quan_data_size() <<
      " vs " << proto->double_csc_data_size() << " ).";
  		  
  LOG(INFO) << " ^_^ here" << " szie: " << proto->csc_quan_data_size();
  proto->clear_double_csc_data();
}



template <>
void Blob<float>::ToProtoPrun(BlobProto* proto, bool write_diff, bool prun, int num, int sparse_diff_num) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  //float* data_vec = cpu_data_prun();
  float* data_vec = mutable_cpu_data();
  int valid_num = 0;

  valid_num = CalWeightPrun(&data_vec, count_, prun, num);
  valid_num = valid_num * 2 + FLAGS_sparse_col + 1;

  if (((FLAGS_sparse_csc && prun /*&& (valid_num < count_)*/) ||
       (FLAGS_sparse_csc && !FLAGS_prun_fc && !FLAGS_prun_conv)) && (sparse_diff_num != 0))
    {
      encode_weight(proto, data_vec, sparse_diff_num);
      // quantization
      if (FLAGS_quan_enable)
	{
	  int valid_num = proto->csc_ptr(proto->csc_ptr_size() - 1);
	  int tmp_idx = 0;
	  float* valid_data;
	  valid_data = (float *)malloc(sizeof(float) * valid_num);
	  assert(valid_data != NULL);

	  for (int idx = 0; idx < proto->csc_data_size(); ++idx)
	    {
	      if (proto->csc_data(idx) != 0)
		{
		  valid_data[tmp_idx] = proto->csc_data(idx);
		  tmp_idx++;
		}
	    }

	  if (tmp_idx != valid_num)
	    LOG(FATAL) << " [Error] Index exceed boundary.( " << tmp_idx << " vs " << valid_num << " )";
	  
	  if (FLAGS_quan_retrain)
	    quan_retrain(proto);
	  else
	    weight_quan(proto, valid_data, valid_num);
	  free(valid_data);
	}
    }
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
void Blob<double>::ToProtoPrun(BlobProto* proto, bool write_diff, bool prun, int num, int sparse_diff_num) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  //double* data_vec = cpu_data_prun();
  double* data_vec = mutable_cpu_data();
  int valid_num = 0;

  valid_num = CalWeightPrun(&data_vec, count_, prun, num);
  valid_num = valid_num * 2 + FLAGS_sparse_col + 1;

  if (((FLAGS_sparse_csc && prun/* && (valid_num < count_)*/) ||
       (FLAGS_sparse_csc && !FLAGS_prun_fc && !FLAGS_prun_conv)) && (sparse_diff_num != 0))
    {
      encode_weight(proto, data_vec, sparse_diff_num);
      
      // quantization
      if (FLAGS_quan_enable)
	{
	  int valid_num = proto->csc_ptr(proto->csc_ptr_size() - 1);
	  int tmp_idx = 0;
	  double* valid_data;
	  valid_data = (double *)malloc(sizeof(double) * valid_num);
	  assert(valid_data != NULL);
	  
	  for (int idx = 0; idx < proto->double_csc_data_size(); ++idx)
	    {
	      if (proto->double_csc_data(idx) != 0)
		{
		  valid_data[tmp_idx] = proto->double_csc_data(idx);
		  tmp_idx++;
		}
	    }
	  
	  if (tmp_idx != valid_num)
	    LOG(FATAL) << " [Error] Index exceed boundary.( " << tmp_idx << " vs " << valid_num << " )";
	  
	  if (FLAGS_quan_retrain)
	    quan_retrain(proto);
	  else
	    weight_quan(proto, valid_data, valid_num);
	  free(valid_data);
	}
    }
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

template <>
void Blob<unsigned int>::decode_weight(const BlobProto* proto, unsigned int** weight) const {
  NOT_IMPLEMENTED;
}

template <>
void Blob<int>::decode_weight(const BlobProto* proto, int** weight) const {
  NOT_IMPLEMENTED;
}

template <>
void Blob<float>::decode_weight(const BlobProto* proto, float** weight) const {
  int row_num = proto->csc_ptr_size();
  float* tmp_data = *weight;
  int base_idx = 0;
  int reality_idx = 0;
  
  CHECK_EQ(FLAGS_sparse_col, (row_num - 1)) << " mismatch data size, need: " <<
    FLAGS_sparse_col << ", reality: " << (row_num - 1);

  for (int i = 0; i < count_; i++)
    tmp_data[i] = 0;

  for(int i = 0; i < FLAGS_sparse_col; i++)
    {
      int start = proto->csc_ptr(i);
      int end = proto->csc_ptr(i+1);
      for (int j = start; j < end; j++)
	{
	  // decode CSC + quantization
	  if (FLAGS_quan_enable && (proto->csc_quan_data_size() != 0))
	    {
	      if (proto->csc_quan_data(reality_idx) == 0)
		{
		  do
		    {
		      base_idx += proto->csc_row_idx(reality_idx);
		      reality_idx++;
		    }while(proto->csc_quan_data(reality_idx) == 0);

		  base_idx += proto->csc_row_idx(reality_idx);
		  tmp_data[base_idx * FLAGS_sparse_col + i] =
		    proto->quan_data(proto->csc_quan_data(reality_idx) - 1);
		  reality_idx++;
		}
	      else
		{
		  base_idx += proto->csc_row_idx(reality_idx);
		  tmp_data[base_idx * FLAGS_sparse_col + i] =
		    proto->quan_data(proto->csc_quan_data(reality_idx) - 1);
		  reality_idx++;
		}
	    }
	  else
	    {
	      // decode CSC
	      if (proto->csc_data(reality_idx) == 0)
		{
		  do
		    {
		      base_idx += proto->csc_row_idx(reality_idx);
		      reality_idx++;
		    }while(proto->csc_data(reality_idx) == 0);

		  base_idx += proto->csc_row_idx(reality_idx);
		  tmp_data[base_idx * FLAGS_sparse_col + i] = proto->csc_data(reality_idx);
		  reality_idx++;
		}
	      else
		{
		  base_idx += proto->csc_row_idx(reality_idx);
		  tmp_data[base_idx * FLAGS_sparse_col + i] = proto->csc_data(reality_idx);
		  reality_idx++;
		}
	    }
	}
    }
}

template <>
void Blob<double>::decode_weight(const BlobProto* proto, double** weight) const {
  int row_num = proto->csc_ptr_size();
  double* tmp_data = *weight;
  int base_idx = 0;
  int reality_idx = 0;

  CHECK_EQ(FLAGS_sparse_col, (row_num - 1)) << " mismatch data size, need: "<<
    FLAGS_sparse_col <<", reality: " << (row_num - 1);

  for (int i = 0; i < count_; i++)
    tmp_data[i] = 0;

  for(int i = 0; i < FLAGS_sparse_col; i++)
    {
      int start = proto->csc_ptr(i);
      int end = proto->csc_ptr(i+1);
      for (int j = start; j < end; j++)
	{
	  // decode CSC + quantization
	  if (FLAGS_quan_enable && (proto->csc_quan_data_size() != 0))
	    {
	      if (proto->csc_quan_data(reality_idx) == 0)
		{
		  do
		    {
		      base_idx += proto->csc_row_idx(reality_idx);
		      reality_idx++;
		    }while(proto->csc_quan_data(reality_idx) == 0);

		  base_idx += proto->csc_row_idx(reality_idx);
		  tmp_data[base_idx * FLAGS_sparse_col + i] =
		    proto->quan_data(proto->csc_quan_data(reality_idx) - 1);
		  reality_idx++;
		}
	      else
		{
		  base_idx += proto->csc_row_idx(reality_idx);
		  tmp_data[base_idx * FLAGS_sparse_col + i] =
		    proto->quan_data(proto->csc_quan_data(reality_idx) - 1);
		  reality_idx++;
		}
	    }
	  else
	    {
	      // decode CSC
	      if (proto->double_csc_data(reality_idx) == 0)
		{
		  do
		    {
		      base_idx += proto->csc_row_idx(reality_idx);
		      reality_idx++;
		    }while(proto->double_csc_data(reality_idx) == 0);

		  base_idx += proto->csc_row_idx(reality_idx);
		  tmp_data[base_idx * FLAGS_sparse_col + i] = proto->double_csc_data(reality_idx);
		  reality_idx++;
		}
	      else
		{
		  base_idx += proto->csc_row_idx(reality_idx);
		  tmp_data[base_idx * FLAGS_sparse_col + i] = proto->double_csc_data(reality_idx);
		  reality_idx++;
		}
	    }
	}
    }
}

template <typename Dtype>
void Blob<Dtype>::weight_quan(BlobProto* proto, Dtype* weight, int num) {
  Dtype max_weight = std::numeric_limits<Dtype>::min();
  Dtype min_weight = std::numeric_limits<Dtype>::max();

  for (int i = 0; i < num; ++i)
    {
      if (max_weight < weight[i])
	max_weight = weight[i];
      if (min_weight > weight[i])
	min_weight = weight[i];
    }

  int centroid_num = 0;
  int* quan_data_num; // share centroid num (a centroid VS. a lost of weights)
  Dtype* quan_data; // centroid
  Dtype* local_last_quan;
  int* quan_label; // for weight index centroid
  Dtype* last_quan_data; // record last iter info
  int* last_quan_label;
  int best_k = 0; //
  Dtype wcss = 0; // WCSS: within-cluster sum of squares
  Dtype last_wcss = std::numeric_limits<Dtype>::max();
  Dtype global_min_wcss = std::numeric_limits<Dtype>::max();

  quan_label = (int *)malloc(sizeof(int) * num);
  assert(quan_label != NULL);
  last_quan_data = (Dtype *)malloc(sizeof(Dtype) * (1 << FLAGS_quan_k_max));
  assert(last_quan_data != NULL);
  last_quan_label = (int *)malloc(sizeof(int) * num);
  assert(last_quan_label != NULL);
  local_last_quan = (Dtype *)malloc(sizeof(Dtype) * (1 << FLAGS_quan_k_max));
  assert(local_last_quan != NULL);

  memset(last_quan_data, 0, sizeof(Dtype) * (1 << FLAGS_quan_k_max));
  memset(last_quan_label, 0, sizeof(int) * num);
  memset(local_last_quan, 0, sizeof(Dtype) * (1 << FLAGS_quan_k_max));

  for (int cluster_num = FLAGS_quan_k_min/*4*/; cluster_num < (FLAGS_quan_k_max+1); ++cluster_num)
    {
      centroid_num = (1 << cluster_num);
      quan_data = (Dtype *)malloc(sizeof(Dtype) * centroid_num);
      assert(quan_data != NULL);
      quan_data_num = (int *)malloc(sizeof(int) * centroid_num);
      assert(quan_data_num != NULL);

      LOG(INFO) << " [Info] cluster: " << cluster_num << " centroid num: " << centroid_num;

      // initial centroid value
      for (int centroid_idx = 0; centroid_idx < centroid_num; ++centroid_idx)
	{
	  quan_data[centroid_idx] = min_weight + (max_weight - min_weight)/centroid_num*centroid_idx;
	  quan_data_num[centroid_idx] = 0;
	}

      // iteration calculate centroid
      int iter_num = 0;
      for (/*int*/ iter_num = 0; iter_num < FLAGS_quan_max_iter; ++iter_num)
	{
	  // use k-means calculate centroid
	  wcss = kmeans(weight, num, &quan_data, &quan_data_num, &quan_label, centroid_num);

	  if (last_wcss == wcss)
	   break;
	  last_wcss = wcss;
	}

      LOG(INFO) << "[Info] iter: " << iter_num-1 << " max iter: " << FLAGS_quan_max_iter;
      LOG(INFO) << " WCSS: " << wcss;

      if (global_min_wcss > wcss)
	{
	  global_min_wcss = wcss;
	  best_k = cluster_num;
	  memset(last_quan_data, 0, sizeof(Dtype) * centroid_num);
	  memset(last_quan_label, 0, sizeof(int) * num);
	  memcpy(last_quan_data, quan_data, sizeof(Dtype) * centroid_num);
	  memcpy(last_quan_label, quan_label, sizeof(int) * num);
	}
    }

  LOG(INFO) << " [Info] best K: " << best_k;

  // write quantization data into blob
  quan_to_blob(proto, last_quan_data, last_quan_label, best_k, num);

  free(quan_data);
  free(quan_label);
  free(last_quan_data);
  free(last_quan_label);
}

template <typename Dtype>
Dtype Blob<Dtype>::kmeans(Dtype* weight, int weight_num, Dtype** data, int** data_num,
			  int** label, int centroid_num){
  Dtype min_distance = std::numeric_limits<Dtype>::max();
  Dtype tmp_distance = 0;
  Dtype wcss = 0;
  int* centroid_idx = *label;
  int* centroid_data_num = *data_num;
  Dtype* centroid_data = *data;

  for (int weight_idx = 0; weight_idx < weight_num; ++weight_idx)
    {
      min_distance = std::numeric_limits<Dtype>::max();
      tmp_distance = 0;
      for (int idx = 0; idx < centroid_num; ++idx)
	{
	  tmp_distance = fabs(weight[weight_idx] - centroid_data[idx]);
	  if (min_distance > tmp_distance)
	    {
	      min_distance = tmp_distance;
	      centroid_idx[weight_idx] = idx;
	    }
	}
    }

  // use the average of every cluster weights update centroid
  for (int idx = 0; idx < centroid_num; ++idx)
    {
      centroid_data[idx] = 0;
      centroid_data_num[idx] = 0;
    }
  for (int weight_idx = 0; weight_idx < weight_num; ++weight_idx)
    {
      centroid_data[centroid_idx[weight_idx]] += weight[weight_idx];
      centroid_data_num[centroid_idx[weight_idx]]++;
    }
  for (int idx = 0; idx < centroid_num; ++idx)
    if (centroid_data_num[idx] != 0)
      centroid_data[idx] = centroid_data[idx]/centroid_data_num[idx];

  // calculate WCSS
  for (int i = 0; i < centroid_num; ++i)
    for (int j = 0; j <= i/*centroid_num*/; ++j)
      wcss += fabs(centroid_data[j] - centroid_data[i]) * fabs(centroid_data[j] - centroid_data[i]);
   
  //for (int weight_idx = 0; weight_idx < weight_num; ++weight_idx)
  //  wcss += fabs(weight[weight_idx] - centroid_data[centroid_idx[weight_idx]]) *
  //    fabs(weight[weight_idx] - centroid_data[centroid_idx[weight_idx]]);
  
  return wcss;
}

template <>
void Blob<int>::quan_to_blob(BlobProto* proto, int* quan_data, int* label, int best_k, int num){
}

template <>
void Blob<unsigned int>::quan_to_blob(BlobProto* proto, unsigned int* quan_data,
				      int* label, int best_k, int num){
}

template <>
void Blob<float>::quan_to_blob(BlobProto* proto, float* quan_data, int* label, int best_k, int num){
  //int label_index = 0;
  int csc_data_idx = 0;
  int max_data_idx = proto->csc_data_size();
  int cluster_num = (1 << best_k);

  for (int cluster_idx = 0; cluster_idx < cluster_num; ++cluster_idx)
    proto->add_quan_data(quan_data[cluster_idx]);
  
  // TODO: rewrite csc_data
  proto->clear_csc_data();
  for (int idx = 0; idx < num; ++idx)
    {
      while(proto->csc_data(csc_data_idx) == 0)
	{
	  proto->add_csc_quan_data(0);
	  csc_data_idx++;
	}

      proto->add_csc_quan_data(label[idx]+1);
      csc_data_idx++;
    }

  if (csc_data_idx != max_data_idx)
    LOG(FATAL) << " [Error] Index exceed boundary( " << csc_data_idx << " vs " << max_data_idx << " ).";
}

template <>
void Blob<double>::quan_to_blob(BlobProto* proto, double* quan_data, int* label, int best_k, int num){
  //int label_index = 0;
  int csc_data_idx = 0;
  int max_data_idx = proto->double_csc_data_size();
  int cluster_num = (1 << best_k);

  for (int cluster_idx = 0; cluster_idx < cluster_num; ++cluster_idx)
    proto->add_double_quan_data(quan_data[cluster_idx]);
  
  // TODO: rewrite csc_data
  proto->clear_double_csc_data();
  for (int idx = 0; idx < num; ++idx)
    {
      while(proto->double_csc_data(csc_data_idx) == 0)
	{
	  proto->add_csc_quan_data(0);
	  csc_data_idx++;
	}

      proto->add_csc_quan_data(label[idx]+1);
      csc_data_idx++;
    }

  if (csc_data_idx != max_data_idx)
    LOG(FATAL) << " [Error] Index exceed boundary( " << csc_data_idx << " vs " << max_data_idx << " ).";
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe
