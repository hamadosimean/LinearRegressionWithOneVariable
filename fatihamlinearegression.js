class fatihamLinearRegression {
  constructor(n_jobs = 1000, lr = 0.001, verbose = 0) {
    this.n_jobs = n_jobs;
    this.lr = lr;
    this.verbose = verbose;
    this.w = 0;
    this.b = 0;
  }

  cost(x, y, w, b) {
    let costSum = 0;
    let m = x.length;
    let f;
    for (let i = 0; i < m; i++) {
      f = w * x[i] + b;
      costSum += Math.pow(f - y[i], 2);
    }

    return (1 / 2) * m * costSum;
  }

  partial_derive(x, y, w, b) {
    let w_b = 0;
    let b_d = 0;
    let f;
    let m = x.length;

    for (let i = 0; i < m; i++) {
      f = w * x[i] + b;
      w_b += (f - y[i]) * x[i];
      b_d += f - y[i];
    }

    return [w_b / m, b_d / m];
  }

  compute_gradient(x, y, w, b) {
    this.cost_history = [];
    let costVal;
    for (let i = 0; i < this.n_jobs; i++) {
      this.cost_history.push(this.cost(x, y, w, b));
      let w_d = this.partial_derive(x, y, w, b)[0];
      let b_d = this.partial_derive(x, y, w, b)[1];

      costVal = this.cost(x, y, w, b);
      w = w - this.lr * w_d;
      b = b - this.lr * b_d;

      if (this.n_jobs < 10000 % 10 == 0) {
        if (this.verbose == 1) {
          console.log(`Iterration ${i} w : ${w} b:${b} cost:${costVal}`);
        }
      }
    }

    return [w, b];
  }

  fit(x, y) {
    let w = 0.000000078887994;
    let b = 0.00000002217956688;
    this.b = this.compute_gradient(x, y, w, b)[1];
    this.w = this.compute_gradient(x, y, w, b)[0];

    return [this.w, this.b];
  }

  predict(x) {
    return x.map((predicted) => {
      return predicted * this.w + this.b;
    });
  }
}

const x = [4, 8, 10, 78, 89, 56];

const y = [4, 8, 10, 78, 89, 56];

const my_reg = new fatihamLinearRegression(
  (n_jobs = 100000),
  (lr = 0.0001),
  (verbose = 0)
);

// console.log(my_reg.cost(x,y,2,3));
// console.log(my_reg.partial_derive(x,y,2,3));

// console.log(my_reg.compute_gradient(x,y,2,5));
// my_reg.compute_gradient(x,y,2,5);
// my_reg.cost_history.forEach((x)=>{
//     console.log(x)
// });

my_reg.fit(x, y);
console.log(`w:${my_reg.w} and b:${my_reg.b}`);

console.log(my_reg.predict([4, 8, 10, 78, 89, 56, 200]));
