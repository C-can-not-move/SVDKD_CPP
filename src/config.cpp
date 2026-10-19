#include "config.h"

namespace svdkd::cfg
{
	RunConfig defaults()
	{
		RunConfig c;
#ifdef _WIN32
		c.data_dir = "D:/DATA/CODE/C++/SVDKD_C++/dataset/cifar-10-batches-bin";
#else
		c.data_dir = "/home/chenzhijiang/userFile/dataset_all/cifar-10-batches-bin";
#endif
		return c;
	}
}  // namespace svdkd::cfg