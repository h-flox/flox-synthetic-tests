import numpy as np


if __name__ == "__main__":
	"""
	These were calculated from the `Async Comparison.ipynb` notebook
    using the results provided in `a3_async_test/out/`.
	"""
	async_idle = np.array([
		0.0362982, 0.0892326, 0.0775805, 0.0541913, 0.0587702, 0.0558669, 
		0.0453912, 0.0704842, 0.0685854, 0.0539671, 0.0667613, 0.0508709,
	])
	
	sync_idle = np.array([
		7.63889, 46.7374, 17.9554, 10.7939, 6.053, 7.31763, 4.40638, 
		23.6189, 13.9019, 4.63037, 17.3608, 17.8858
	])
	
	out_template = "[{}] :: mean={:0.5f}  |  std={:0.5f}"
	print(out_template.format(
		"sync", sync_idle.mean(), sync_idle.std()
	))
	print(out_template.format(
		"async", async_idle.mean(), async_idle.std()
	))

	
