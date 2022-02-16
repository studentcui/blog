---
title: ACM模板
date: 2019-11-26 22:54:04
tags:
 - ACM
categories: 
 - other
---

[TOC]

# 1 字符串处理

## 1.1 KMP

```c++
/*
 *Next[]含义：x[i-Next[i]...i-1]=x[0...Next[i]-1]
 *Next[i]为满足x[i-z...i-1]=x[0...z-1]的最大z值（就是x的自身匹配）
 */
void getNext(char x[], int m, int Next[]){
    int i, j;
    j = Next[0] = -1;
    i = 0;
    while(i<m){
        while(-1 != j && x[i] != x[j]) j = Next[j];
        if(x[++i] == x[++j]) Next[i] = Next[j];
        else Next[i] = j;
    }
}
/*
 *返回x在y中出现的次数，可以重叠
 */
int Next[10010];
int kmp(char x[], int m, char y[], int n){ //x模式串y主串
    int i, j;
    int ans = 0;
    getNext(x,m,Next);
    i = j = 0;
    while(i < n){
        while(-1 != j && y[i] != x[j]) j = Next[j];
        i++;j++;
        if(j >= m){
            ans++;
            j = Next[j];
        }
    }
    return ans;
} 
```

## 1.2 扩展KMP

```c++
//Next[i]:x[i...m-1] 与x[0...m-1] 的最长公共前缀
//extend[i]:y[i...n-1] 与x[0...m-1] 的最长公共前缀
void preEKMP(char x[],int m,int Next[]) {
    Next[0] = m;
    int j = 0;
    while( j+1 < m && x[j] == x[j+1] )
        j++;
    Next[1] = j;
    int k = 1;
    for(int i = 2; i < m; i++) {
        int p = Next[k]+k−1;
        int L = Next[i−k];
        if( i+L < p+1 )
            Next[i] = L;
        else {
            j = max(0,p−i+1);
            while( i+j < m && x[i+j] == x[j])
                j++;
            Next[i] = j;
            k = i;
        }
    }
}
void EKMP(char x[],int m,char y[],int n,int Next[],int extend[]) {
    pre_EKMP(x,m,Next);
    int j = 0;
    while(j < n && j < m && x[j] == y[j])
        j++;
    extend[0] = j;
    int k = 0;
    for(int i = 1; i < n; i++) {
        int p = extend[k]+k−1;
        int L = Next[i−k];
        if(i+L < p+1)
            extend[i] = L;
        else {
            j = max(0,p−i+1);
            while( i+j < n && j < m && y[i+j] == x[j] )
                j++;
            extend[i] = j;
            k = i;
        }
    }
}
```



# 2 数学

## 2.1 素数

### 2.1.1 素数筛选（判断<MAXN的数是否素数）

```c++
/*
* 素数筛选，判断小于MAXN 的数是不是素数。
* notprime 是一张表，为false 表示是素数，true 表示不是素数
*/
const int MAXN=1000010;
bool notprime[MAXN];//值为false 表示素数，值为true 表示非素数
void init() {
    memset(notprime,false,sizeof(notprime));
    notprime[0]=notprime[1]=true;
    for(int i=2; i<MAXN; i++)
        if(!notprime[i]) {
            if(i>MAXN/i)
                continue;//防止后面i*i 溢出(或者i,j 用long long)
            //直接从i*i 开始就可以，小于i 倍的已经筛选过了, 注意是j+=i
            for(int j=i*i; j<MAXN; j+=i)
                notprime[j]=true;
        }
}
```

### 2.1.2 素数筛选（筛选出<=MAXN的素数）

```c++
/*
* 素数筛选，存在小于等于MAXN 的素数
* prime[0] 存的是素数的个数
*/
const int MAXN=10000;
int prime[MAXN+1];
void getPrime() {
    memset(prime,0,sizeof(prime));
    for(int i=2; i<=MAXN; i++) {
        if(!prime[i])
            prime[++prime[0]]=i;
        for(int j=1; j<=prime[0]&&prime[j]<=MAXN/i; j++) {
            prime[prime[j]*i]=1;
            if(i%prime[j]==0)
                break;
        }
    }
}
```

## 2.2 幂

### 2.2.1 快速幂

```c++
typedef long long ll;
ll mod;
ll ksm(ll a, ll n) { //计算a^n % mod
    ll re = 1;
    while(n) {
        if(n & 1)//判断n的最后一位是否为1
            re = (re * a) % mod;
        n >>= 1;//舍去n的最后一位
        a = (a * a) % mod;//将a平方
    }
    return re % mod;
}
```

### 2.2.2 矩阵快速幂

```c++
typedef long long Type;
const int MAXM = 10, mod = 10; //MAXM方阵边长 mod取模
struct Matrix {
    Type a[MAXM][MAXM] ;
    Type* operator [] (const int id) const {
        return (Type*)(a[id]);
    }
    Matrix(bool op = false) {
        for(int i = 0; i < MAXM; i++)
            for(int j = 0; j < MAXM; j++)
                a[i][j] = op * ( i == j ) ;
    }
    Matrix operator * (const Matrix A) const {
        Matrix ans ;
        for ( int i = 0; i < MAXM; i++)
            for ( int j = 0; j < MAXM; j++)
                for ( int k = 0; k < MAXM; k++)
                    ans[i][j] = (ans[i][j] + a[i][k] * A[k][j]) % mod;
        return ans;
    }
    Matrix operator ^ (int mi) const {
        Matrix ans(true), A = (*this);
        while (mi) {
            if (mi & 1)
                ans = ans * A;
            A = A * A;
            mi >>= 1;
        }
        return ans;
    }
};
```

# 3 其他

## 3.1 STL

### 3.1.1 priority_queue优先队列（堆）

```c++
#include<queue>
//方法一
priority_queue<int> q;   　　　　　　　　　　　　  //通过操作，按照元素从大到小的顺序出队
priority_queue<int,vector<int>, greater<int> > q;  　　//通过操作，按照元素从小到大的顺序出队
//方法二
struct cmp {
    operator bool()(int x, int y) {
        return x > y; //小的优先级高 因为优先出列判定为!cmp 所以反向定义实现小值优先
    }
};
priority_queue<int, vector<int>, cmp> q;
//方法三
struct node {
    int x;
    friend bool operator < (node a, node b) {
        return a.x > b.x;//结构体中，x小的优先级高
    }
};
priority_queue<node>q;
```

## 3.2 排序

### 3.2.1 归并排序

```c++
const int MAXN = 10;
int a[MAXN], t[MAXN]; //a为排序的数组，t为临时数组
void merge_sort(int x, int y) { //用法merge_sort(0,n)
    if(y-x<=1)
        return;
    int m = x+(y-x)/2;
    int p = x, q = m, i = x;
    merge_sort(x,m);
    merge_sort(m,y);
    while(p < m || q < y) {
        if(q >= y || (p < m && a[p] < a[q]))
            t[i++] = a[p++];
        else
            t[i++] = a[q++];
    }
    for(int i = x; i < y; ++i)
        a[i] = t[i];
}
```

# 4 图论

## 4.1 最短路

### 4.1.1 Floyd

```c++
for(int k = 1; k <= n; ++k)
    for(int i = 1; i <= n; ++i)
        for(int j = 1; j <= n; ++j)
            if(e[i][j] > e[i][k]+e[k][j])
                e[i][j] = e[i][k]+e[k][j];
```

### 4.1.2 Dijkstra单源最短路

```c++
int e[MAXN][MAXN], dis[MAXN], book[MAXN], i, j, n, m, u, v, min;
int inf = 0x3f3f3f3f;
//初始化
for(i = 1; i <= n; ++i)
    for(j = 1; j <= n; ++j)
        e[i][j] = i==j?0:inf;
//读入边
//初始化dis数组，一号顶点到其余顶点的初始距离
for(i = 1; i <= n; ++i)
    dis[i] = e[1][i];
//book数组初始化
memset(book, 0, sizeof book);
book[1] = 1;
//dijkstra核心语句
for(i = 1; i <= n-1; ++i){
    min = inf;
    for(j = 1; j <= n; ++j){
        if(book[j] == 0 && dis[j] < min){
            min = dis[j];
            u = j;
        }
    }
    book[u] = 1;
    for(v = 1; v <= n; ++v)
        if(e[u][v] < inf)
            if(dis[v] > dis[u]+e[u][v])
                dis[v] = dis[u]+e[u][v];
}
```

### 4.1.3 Bellman-Ford（负权边）

```c++
int dis[10],bak[10],i,k,n,m,u[10],v[10],w[10],check,flag;
int inf = 0x3f3f3f3f;
//n顶点数m边数
//读入边
//初始化dis
for(i = 1; i <= n; ++i)
    dis[i] = inf;
dis[1] = 0;
//Bellman-Ford核心
for(k = 1; k <= n-1; ++k){
    for(i = 1; i <= n; ++i) bak[i] = dis[i];
    for(i = 1; i <= m; ++i)
        if(dis[v[i]] > dis[u[i]] + w[i])
            dis[v[i]] = dis[u[i]] + w[i];
    check = 0;
    for(i = 1; i <= n; ++i)
        if(bak[i] != dis[i]){
            check = 1;
            break;
        }
    if(check == 0) break; //如果dis没更新，提前退出
}
//检测负权回路
flag = 0; 
for(i = 1; i <= m; ++i)
    if(dis[v[i]] > dis[u[i]] + w[i]) flag = 1;
if(flag = 1) puts("存在负权回路");
```

### 4.1.4 SPFA 单源最短路

```c++
/*
 * 单源最短路SPFA
 * 时间复杂度0(kE)
 * 这个是队列实现，有时候改成栈实现会更加快，很容易修改
 * 这个复杂度是不定的
 */
const int MAXN=1010;
const int INF=0x3f3f3f3f;
struct Edge {
    int v;
    int cost;
    Edge(int _v=0,int _cost=0):v(_v),cost(_cost) {}
};
vector<Edge>E[MAXN];
void addedge(int u,int v,int w) {
    E[u].push_back(Edge(v,w));
}
bool vis[MAXN];//在队列标志
int cnt[MAXN];//每个点的入队列次数
int dist[MAXN];
bool SPFA(int start,int n) {
    memset(vis,false,sizeof(vis));
    for(int i=1; i<=n; i++)
        dist[i]=INF;
    vis[start]=true;
    dist[start]=0;
    queue<int>que;
    while(!que.empty())
        que.pop();
    que.push(start);
    memset(cnt,0,sizeof(cnt));
    cnt[start]=1;
    while(!que.empty()) {
        int u=que.front();
        que.pop();
        vis[u]=false;
        for(int i=0; i<E[u].size(); i++) {
            int v=E[u][i].v;
            if(dist[v]>dist[u]+E[u][i].cost) {
                dist[v]=dist[u]+E[u][i].cost;
                if(!vis[v]) {
                    vis[v]=true;
                    que.push(v);
                    if(++cnt[v]>n)
                        return false; //cnt[i] 为入队列次数，用来判定是否存在负环回路
                }
            }
        }
    }
    return true;
}
```

## 4.2 最小生成树

### 4.2.1 Kruskal

边排序，两端不在同一个子树则连接，稀疏图

```c++
const int MAXN=110;//最大点数
const int MAXM=10000;//最大边数
int F[MAXN];//并查集使用
struct Edge {
    int u,v,w;
} edge[MAXM]; //存储边的信息，包括起点/终点/权值
int tol;//边数，加边前赋值为0
void addedge(int u,int v,int w) {
    edge[tol].u=u;
    edge[tol].v=v;
    edge[tol++].w=w;
}
//排序函数，讲边按照权值从小到大排序
bool cmp(Edge a,Edge b) {
    return a.w<b.w;
}
int find(int x) {
    if(F[x]==−1)
        return x;
    else
        return F[x]=find(F[x]);
}
//传入点数，返回最小生成树的权值，如果不连通返回-1
int Kruskal(int n) {
    memset(F,−1,sizeof(F));
    sort(edge,edge+tol,cmp);
    int cnt=0;//计算加入的边数
    int ans=0;
    for(int i=0; i<tol; i++) {
        int u=edge[i].u;
        int v=edge[i].v;
        int w=edge[i].w;
        int t1=find(u);
        int t2=find(v);
        if(t1!=t2) {
            ans+=w;
            F[t1]=t2;
            cnt++;
        }
        if(cnt==n−1)
            break;
    }
    if(cnt<n−1)
        return −1;//不连通
    else
        return ans;
}
```

### 4.2.2 Prim

与生成树最近的顶点加入生成树

```c++
/*
* Prim 求MST
* 耗费矩阵cost[][]，标号从0 开始，0∼n-1
* 返回最小生成树的权值，返回-1 表示原图不连通
*/
const int INF=0x3f3f3f3f;
const int MAXN=110;
bool vis[MAXN];
int lowc[MAXN];
//点是0 n-1
int Prim(int cost[][MAXN],int n) {
    int ans=0;
    memset(vis,false,sizeof(vis));
    vis[0]=true;
    for(int i=1; i<n; i++)
        lowc[i]=cost[0][i];
    for(int i=1; i<n; i++) {
        int minc=INF;
        int p=−1;
        for(int j=0; j<n; j++)
            if(!vis[j]&&minc>lowc[j]) {
                minc=lowc[j];
                p=j;
            }
        if(minc==INF)
            return −1;//原图不连通
        ans+=minc;
        vis[p]=true;
        for(int j=0; j<n; j++)
            if(!vis[j]&&lowc[j]>cost[p][j])
                lowc[j]=cost[p][j];
    }
    return ans;
}
```



# 5 数据结构

## 5.1 RMQ

### 5.1.1 一维

```c++
//求最大值，数组下标从1 开始。
//求最小值，或者最大最小值下标，或者数组从0 开始对应修改即可。
const int MAXN = 50010;
int dp[MAXN][20];
int mm[MAXN];
//初始化RMQ, b 数组下标从1 开始，从0 开始简单修改
void initRMQ(int n,int b[]) {
    mm[0] = −1;
    for(int i = 1; i <= n; i++) {
        mm[i] = ((i&(i−1)) == 0)?mm[i−1]+1:mm[i−1];
        dp[i][0] = b[i];
    }
    for(int j = 1; j <= mm[n]; j++)
        for(int i = 1; i + (1<<j) −1 <= n; i++)
            dp[i][j] = max(dp[i][j−1],dp[i+(1<<(j−1))][j−1]);
}
//查询最大值
int rmq(int x,int y) {
    int k = mm[y−x+1];
    return max(dp[x][k],dp[y−(1<<k)+1][k]);
}
```

## 5.2 线段树

```c++
typedef long long ll;
unsigned ll n,m,a[MAXN],ans[MAXN<<2],tag[MAXN<<2];
inline void push_up(ll p){
	ans[p] = ans[p<<1] + ans[p<<1|1];
}
void build(ll p, ll l, ll r){
    tag[p] = 0;
    if(l == r){
        ans[p] = a[l];
        return;
    }
    ll mid = (l+r)>>1;
    build(p<<1,l,mid);
    build(p<<1|1,mid+1,r);
    push_up(p);
}
inline void f(ll p, ll l, ll r, ll k){
    tag[p] = tag[p]+k;
    ans[p] = ans[p]+k*(r-l+1);
}
inline void push_down(ll p, ll l, ll r){
    ll mid = (l+r)>>1;
    f(p<<1,l,mid,tag[p]);
    f(p<<1|1,mid+1,r,tag[p]);
    tag[p] = 0;
}
inline void update(ll nl, ll nr, ll l, ll r, ll p, ll k){
    if(nl <= l && r <= nr){
        ans[p] += k*(r-l+1);
        tag[p] += k;
        return;
    }
    push_down(p,l,r);
    ll mid = (l+r)>>1;
    if(nl <= mid) update(nl, nr, l, mid, p<<1, k);
    if(nr > mid) update(nl, nr, mid+1, r, p<<1|1, k);
    push_up(p);
}
ll query(ll qx, ll qy, ll l, ll r, ll p){
	ll res = 0;
    if(qx <= mid) res += query(qx,qy,l,mid,p<<1);
    if(qy > mid) res += query(qx,qy,mid+1,r,p<<1|1);
    return res;
}
//main{build(1,1,n)}
```

