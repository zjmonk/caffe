// Initial version
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SemiPairLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  int dim = bottom[0]->count()/bottom[0]->num();
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  printf("count: %d --- num: %d --- dim: %d\n", count, num, dim);
  CHECK_EQ(bottom[0]->channels(), dim);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  
  // S matrix is n*n
  simi_s.Reshape(num,num,1,1);
  theta.Reshape(num,num,1,1);
}

template <typename Dtype>
void SemiPairLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss(0.0);
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count/num;
  //printf("count: %d --- num: %d --- dim: %d\n", count, num, dim);
  const Dtype* bottom_data = bottom[0]->cpu_data(); // bottom_data is n*dim matrix
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* simi_s_data = simi_s.mutable_cpu_data();
  Dtype* theta_data = theta.mutable_cpu_data();
  // Calculate S matrix online
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < num; j++) {
      if (bottom_label[i] == bottom_label[j])
         simi_s_data[i*num+j] = Dtype(1);
      else
         simi_s_data[i*num+j] = Dtype(0);
    }
  }
  // Calculate loss
  // theta[i,j] = 1/2*(bi.*bj)
  // l = sum(sij*theta[i,j]-log(1+exp(theta[i,j])))
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < num; j++) {
       if (simi_s_data[i*num+j] == Dtype(1)) {
            theta_data[i*num + j] = Dtype(0.5)*caffe_cpu_dot(dim, bottom_data+i*dim, bottom_data+j*dim);
            loss = loss - (simi_s_data[i*num+j]*theta_data[i*num+j] - log(Dtype(1)+exp(theta_data[i*num+j])));
       }
    }
  }
  loss = loss / static_cast<Dtype>(num);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SemiPairLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* simi_s_data = simi_s.cpu_data();
    const Dtype* theta_data = theta.cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    int dim = count/num;
    Dtype formerPart = 0;
    Dtype latterPart = 0;
    Dtype one = Dtype(1);
    Dtype mone = Dtype(-1);
    // Back propagation for each image
    // b[i,j] = 0.5*(sum(1/(1+exp(-theta[i,j])-s[i,j])*u(j)+sum(1/(1+exp(-theta[j,i])))
    for (int i = 0; i<num; i++){
        for (int k = 0; k<dim; k++) {
            for (int j =0; j<num; j++){
               if (simi_s_data[i*num+j] == 1) {
                   formerPart = bottom_data[j*dim+k]*(one/(one+exp(mone*(theta_data[i*num+j]))) - simi_s_data[i*num+j]);
                   latterPart = bottom_data[j*dim+k]*(one/(one+exp(mone*(theta_data[j*num+i]))) - simi_s_data[j*num+i]);
                   bottom_diff[i*dim+k] += Dtype(0.5)*(formerPart+latterPart);
               }
            }
        }
    }

    //const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, top[0]->cpu_diff()[0]/Dtype(num), bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(SemiPairLossLayer);
#endif

INSTANTIATE_CLASS(SemiPairLossLayer);
REGISTER_LAYER_CLASS(SemiPairLoss);

}  // namespace caffe
