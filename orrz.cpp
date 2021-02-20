#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <queue>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using base_type_t = int64_t;

double get_time();

template <typename T>
using edge_list_t = std::vector<std::tuple<uint32_t, uint32_t, T, bool>>;

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
    o << "[";
    for(size_t i = 0; i < v.size(); i++) {
        o << v[i];
        if(i != v.size() - 1) {
            o << ", ";
        }
    }
    o << "]";
    return o;
}

template <typename T>
class Graph {
  public:
    std::vector<uint32_t> adj;
    std::vector<uint32_t> offset;
    std::vector<T> value;
    std::vector<bool> strict;
    std::vector<uint32_t> in_degree;
    std::vector<uint32_t> start;

    Graph(uint32_t n, const edge_list_t<T>& edges) {
        using namespace std;
        uint32_t m = edges.size();
        adj.resize(m);
        value.resize(m);
        strict.resize(m);
        offset.resize(n + 1);
        in_degree.resize(n);
        for(uint32_t i = 0; i < m; i++) {
            offset[get<0>(edges[i])]++;
            in_degree[get<1>(edges[i])]++;
        }
        uint32_t acc = 0;
        for(uint32_t i = 0; i < n + 1; i++) {
            uint32_t tmp = offset[i];
            offset[i] = acc;
            acc += tmp;
        }
        vector<uint32_t> tmp = offset;
        for(uint32_t i = 0; i < m; i++) {
            uint32_t idx = get<0>(edges[i]);
            adj[tmp[idx]] = get<1>(edges[i]);
            value[tmp[idx]] = get<2>(edges[i]);
            strict[tmp[idx]] = get<3>(edges[i]);
            tmp[idx]++;
        }
        // https://core.ac.uk/download/pdf/54846862.pdf (Gabow, 2000)
        vector<uint32_t> label(n);
        stack<uint32_t> S, B;
        uint32_t c = n;
        function<void(uint32_t)> scc = [&scc, this, &c, &S, &B, &label](uint32_t node) {
            S.push(node);
            label[node] = S.size();
            B.push(label[node]);
            for(size_t i = this->offset[node]; i < this->offset[node + 1]; i++) {
                if(!label[this->adj[i]]) {
                    scc(this->adj[i]);
                } else {
                    while(B.top() > label[this->adj[i]]) {
                        B.pop();
                    }
                }
            }
            if(B.top() == label[node]) {
                B.pop();
                c++;
                while(S.size() >= label[node]) {
                    uint32_t w = S.top();
                    S.pop();
                    label[w] = c;
                }
            }
        };
        for(size_t i = 0; i < n; i++) {
            if(!label[i]) {
                scc(i);
            }
        }
        vector<bool> zeroin(true, c - n);
        size_t j = 0;
        for(size_t i = 0; i < num_edges(); i++) {
            while(j < num_nodes() && offset[j + 1] == i) {
                j++;
            }
            if(label[j] != label[adj[i]]) {
                zeroin[label[adj[i]] - n - 1] = false;
            }
        }
        for(size_t i = 0; i < n; i++) {
            if(zeroin[label[i] - n - 1]) {
                zeroin[label[i] - n - 1] = false;
                start.push_back(i);
            }
        }
    }

    size_t num_nodes() const { return in_degree.size(); }

    size_t num_edges() const { return adj.size(); }
};

template <typename T>
class EpsilonNumber {
  public:
    T real;
    T epsilon;

    EpsilonNumber(T _real = T(0), T _epsilon = T(0)) : real(_real), epsilon(_epsilon) {}

    EpsilonNumber<T> operator+(const EpsilonNumber<T>& other) const {
        return EpsilonNumber<T>(real + other.real, epsilon + other.epsilon);
    }

    EpsilonNumber<T> operator-(const EpsilonNumber<T>& other) const {
        return EpsilonNumber<T>(real - other.real, epsilon - other.epsilon);
    }

    EpsilonNumber<T> operator+=(const EpsilonNumber<T>& other) {
        real += other.real;
        epsilon += other.epsilon;
        return *this;
    }

    EpsilonNumber<T> operator/(const EpsilonNumber<T>& other) const {
        assert(other.epsilon == T(0));
        return EpsilonNumber<T>(real / other.real, epsilon / other.real);
    }

    bool operator<(const EpsilonNumber<T>& other) const {
        if(real != other.real) {
            return real < other.real;
        }
        return epsilon < other.epsilon;
    }

    bool operator<=(const EpsilonNumber<T>& other) const {
        if(real != other.real) {
            return real < other.real;
        }
        return epsilon <= other.epsilon;
    }

    bool operator==(const EpsilonNumber<T>& other) const { return real == other.real && epsilon == other.epsilon; }
};

namespace std {
template <typename T>
class numeric_limits<EpsilonNumber<T>> {
  public:
    static EpsilonNumber<T> max() { return EpsilonNumber<T>(numeric_limits<T>::max(), numeric_limits<T>::max()); };
};
} // namespace std

template <typename T>
std::ostream& operator<<(std::ostream& o, const EpsilonNumber<T>& v) {
    if(v.real != T(0) && v.epsilon != T(0)) {
        o << "(";
    }
    if(v.real != T(0) || v.epsilon == T(0)) {
        o << v.real;
    }
    if(v.epsilon != T(0)) {
        if(v.epsilon > T(0) && v.real != T(0)) {
            o << "+";
        }
        o << v.epsilon << "ε";
    }
    if(v.real != T(0) && v.epsilon != T(0)) {
        o << ")";
    }
    return o;
}

template <typename T>
void canonize(Graph<T>& g) {}

template <typename T>
void canonize(Graph<EpsilonNumber<T>>& g) {
    for(size_t i = 0; i < g.num_edges(); i++) {
        if(g.strict[i]) {
            g.value[i] += EpsilonNumber<T>(T(0), T(-1));
        }
    }
}

template <typename T>
std::istream& operator>>(std::istream& i, EpsilonNumber<T>& v) {
    T x;
    i >> x;
    v = EpsilonNumber<T>(x);
    return i;
}

// Bellman-Ford + Time out
template <typename T>
std::optional<std::vector<T>> BFTO(const Graph<T>& g) {
    using namespace std;
    size_t iters = 0;
    size_t updates = 0;
    vector<T> distance(g.num_nodes(), numeric_limits<T>::max() / T(2));
    for(uint32_t node : g.start) {
        distance[node] = T(0);
    }
    for(uint32_t i = 0; i < g.num_nodes() - 1; i++) {
        iters++;
        bool updated = false;
        uint32_t j = 0;
        for(uint32_t k = 0; k < g.num_edges(); k++) {
            while(j < g.num_nodes() && g.offset[j + 1] == k) {
                j++;
            }
            if(distance[j] + g.value[k] < distance[g.adj[k]]) {
                updates++;
                distance[g.adj[k]] = distance[j] + g.value[k];
                updated = true;
            }
        }
        if(!updated) {
            break;
        }
    }
    cerr << "BFTO_ITERS: " << iters << endl;
    cerr << "BFTO_UPDAT: " << updates << endl;
    uint32_t j = 0;
    for(uint32_t k = 0; k < g.num_edges(); k++) {
        while(j < g.num_nodes() && g.offset[j + 1] == k) {
            j++;
        }
        if(distance[j] + g.value[k] < distance[g.adj[k]]) {
            // To make a certificate find a cycle in the parent graph
            return {};
        }
    }
    return distance;
}

// Bellman-Ford-Moore + Tarjan's subtree disassembly
template <typename T>
std::optional<std::vector<T>> BFCT(const Graph<T>& g) {
    using namespace std;
    size_t iters = 0;
    size_t updates = 0;
    const uint32_t NONE = uint32_t(-1);
    const T MAX = numeric_limits<T>::max() / T(2);
    size_t n = g.num_nodes();
    vector<uint32_t> parent(n, NONE), first_child(n, NONE), left_sibling(n, NONE), right_sibling(n, NONE);
    vector<T> distance(n, MAX);
    queue<uint32_t> q;
    stack<uint32_t> s;
    for(uint32_t node : g.start) {
        distance[node] = T(0);
        q.push(node);
    }
    while(!q.empty()) {
        iters++;
        uint32_t node = q.front();
        q.pop();
        if(distance[node] == MAX) {
            continue;
        }
        for(size_t i = g.offset[node]; i < g.offset[node + 1]; i++) {
            if(distance[node] + g.value[i] < distance[g.adj[i]]) {
                updates++;
                distance[g.adj[i]] = distance[node] + g.value[i];
                q.push(g.adj[i]);
                if(first_child[g.adj[i]] != NONE) {
                    s.push(first_child[g.adj[i]]);
                    first_child[g.adj[i]] = NONE;
                }
                while(!s.empty()) {
                    uint32_t child = s.top();
                    s.pop();
                    if(child == node) {
                        cerr << "BFCT_ITERS: " << iters << endl;
                        cerr << "BFCT_UPDAT: " << updates << endl;
                        // To make a certificate generate the node stack
                        return {};
                    }
                    parent[child] = NONE;
                    distance[child] = MAX;
                    left_sibling[child] = NONE;
                    if(right_sibling[child] != NONE) {
                        s.push(right_sibling[child]);
                    }
                    right_sibling[child] = NONE;
                    if(first_child[child] != NONE) {
                        s.push(first_child[child]);
                    }
                    first_child[child] = NONE;
                }
                if(parent[g.adj[i]] != NONE) {
                    if(left_sibling[g.adj[i]] != NONE) {
                        right_sibling[left_sibling[g.adj[i]]] = right_sibling[g.adj[i]];
                    }
                    if(right_sibling[g.adj[i]] != NONE) {
                        left_sibling[right_sibling[g.adj[i]]] = left_sibling[g.adj[i]];
                    }
                    if(first_child[parent[g.adj[i]]] == g.adj[i]) {
                        first_child[parent[g.adj[i]]] = right_sibling[g.adj[i]];
                    }
                }
                parent[g.adj[i]] = node;
                if(first_child[node] != NONE) {
                    left_sibling[first_child[node]] = g.adj[i];
                }
                right_sibling[g.adj[i]] = first_child[node];
                left_sibling[g.adj[i]] = NONE;
                first_child[node] = g.adj[i];
            }
        }
    }
    cerr << "BFCT_ITERS: " << iters << endl;
    cerr << "BFCT_UPDAT: " << updates << endl;
    return distance;
}

// Goldberg-Radzik + Admissible graph search
template <typename T>
std::optional<std::vector<T>> GORC(const Graph<T>& g) {
    using namespace std;
    size_t iters = 0;
    size_t updates = 0;
    vector<uint32_t> A, B;
    vector<T> distance(g.num_nodes(), numeric_limits<T>::max() / T(2));
    for(uint32_t node : g.start) {
        distance[node] = T(0);
        B.push_back(node);
    }
    vector<bool> visited(g.num_nodes());
    vector<uint32_t> label(g.num_nodes());
    function<void(uint32_t)> toposort = [&toposort, &g, &A, &visited, &distance](uint32_t node) {
        visited[node] = true;
        for(size_t i = g.offset[node]; i < g.offset[node + 1]; i++) {
            if(!visited[g.adj[i]] && g.value[i] + distance[node] - distance[g.adj[i]] <= T(0)) {
                toposort(g.adj[i]);
            }
        }
        A.push_back(node);
    };
    while(!B.empty()) {
        iters++;
        fill(visited.begin(), visited.end(), false);
        for(auto node : B) {
            if(!visited[node]) {
                bool ok = false;
                for(size_t i = g.offset[node]; i < g.offset[node + 1] && !ok; i++) {
                    if(g.value[i] + distance[node] - distance[g.adj[i]] < T(0)) {
                        ok = true;
                    }
                }
                if(ok) {
                    toposort(node);
                }
            }
        }
        reverse(A.begin(), A.end());
        B.clear();
        for(auto node : A) {
            for(size_t i = g.offset[node]; i < g.offset[node + 1]; i++) {
                if(distance[node] + g.value[i] < distance[g.adj[i]]) {
                    updates++;
                    distance[g.adj[i]] = distance[node] + g.value[i];
                    B.push_back(g.adj[i]);
                }
            }
        }
        // https://core.ac.uk/download/pdf/54846862.pdf (Gabow, 2000)
        fill(label.begin(), label.end(), 0);
        stack<uint32_t> S, BB;
        uint32_t c = g.num_nodes();
        function<void(uint32_t)> scc = [&scc, &g, &c, &S, &BB, &label, &distance](uint32_t node) {
            S.push(node);
            label[node] = S.size();
            BB.push(label[node]);
            for(size_t i = g.offset[node]; i < g.offset[node + 1]; i++) {
                if(g.value[i] + distance[node] - distance[g.adj[i]] <= T(0)) {
                    if(!label[g.adj[i]]) {
                        scc(g.adj[i]);
                    } else {
                        while(BB.top() > label[g.adj[i]]) {
                            BB.pop();
                        }
                    }
                }
            }
            if(BB.top() == label[node]) {
                BB.pop();
                c++;
                while(S.size() >= label[node]) {
                    uint32_t w = S.top();
                    S.pop();
                    label[w] = c;
                }
            }
        };
        for(size_t i = 0; i < A.size(); i++) {
            if(!label[A[i]]) {
                scc(A[i]);
            }
        }
        size_t j = 0;
        for(size_t i = 0; i < g.num_edges(); i++) {
            while(j < g.num_nodes() && g.offset[j + 1] == i) {
                j++;
            }
            if(label[j] && label[j] == label[g.adj[i]] && g.value[i] + distance[j] - distance[g.adj[i]] < T(0)) {
                cerr << "GORC_ITERS: " << iters << endl;
                cerr << "GORC_UPDAT: " << updates << endl;
                // To make a certificate find a negative cycle in this SCC
                return {};
            }
        }
        A.clear();
    }
    cerr << "GORC_ITERS: " << iters << endl;
    cerr << "GORC_UPDAT: " << updates << endl;
    return distance;
}

template <typename T>
std::optional<std::vector<EpsilonNumber<T>>> ORRZ(const Graph<T>& g, const std::vector<T>& distance) {
    using namespace std;
    size_t n = g.num_nodes();
    vector<EpsilonNumber<T>> final(n);
    for(size_t i = 0; i < n; i++) {
        final[i].real = distance[i];
    }
    vector<uint32_t> label(n);
    stack<uint32_t> S, B;
    uint32_t c = g.num_nodes();
    // https://core.ac.uk/download/pdf/54846862.pdf (Gabow, 2000)
    function<void(uint32_t)> scc = [&scc, &g, &c, &S, &B, &label, &distance](uint32_t node) {
        S.push(node);
        label[node] = S.size();
        B.push(label[node]);
        for(size_t i = g.offset[node]; i < g.offset[node + 1]; i++) {
            if(g.value[i] + distance[node] - distance[g.adj[i]] == T(0)) {
                if(!label[g.adj[i]]) {
                    scc(g.adj[i]);
                } else {
                    while(B.top() > label[g.adj[i]]) {
                        B.pop();
                    }
                }
            }
        }
        if(B.top() == label[node]) {
            B.pop();
            c++;
            while(S.size() >= label[node]) {
                uint32_t w = S.top();
                S.pop();
                label[w] = c;
            }
        }
    };
    double start = get_time();
    for(size_t i = 0; i < n; i++) {
        if(!label[i]) {
            scc(i);
        }
    }
    cerr << "SCC: " << get_time() - start << endl;
    start = get_time();
    size_t j = 0;
    for(size_t i = 0; i < g.num_edges(); i++) {
        while(j < g.num_nodes() && g.offset[j + 1] == i) {
            j++;
        }
        if(label[j] == label[g.adj[i]] && g.value[i] + distance[j] - distance[g.adj[i]] == T(0) && g.strict[i]) {
            // To make a certificate find a negative cycle in this SCC that includes this edge
            return {};
        }
    }
    cerr << "RED: " << get_time() - start << endl;
    start = get_time();
    vector<uint32_t> order;
    vector<bool> visited(n);
    order.reserve(n);
    function<void(uint32_t)> toposort = [&toposort, &g, &order, &visited, &distance](uint32_t node) {
        visited[node] = true;
        for(size_t i = g.offset[node]; i < g.offset[node + 1]; i++) {
            if(!visited[g.adj[i]] && g.value[i] + distance[node] - distance[g.adj[i]] == T(0)) {
                toposort(g.adj[i]);
            }
        }
        order.push_back(node);
    };
    for(size_t i = 0; i < n; i++) {
        if(!visited[i]) {
            toposort(i);
        }
    }
    reverse(order.begin(), order.end());
    cerr << "TOP: " << get_time() - start << endl;
    start = get_time();
    for(auto node : order) {
        for(size_t i = g.offset[node]; i < g.offset[node + 1]; i++) {
            if(g.value[i] + distance[node] - distance[g.adj[i]] == T(0)) {
                final[g.adj[i]].epsilon = min(final[g.adj[i]].epsilon, final[node].epsilon - (g.strict[i] ? T(1) : T(0)));
            }
        }
    }
    cerr << "TCH: " << get_time() - start << endl;
    return final;
}

template <typename T>
bool validate(const Graph<EpsilonNumber<T>>& g, const std::vector<EpsilonNumber<T>>& certificate, bool success) {
    using namespace std;
    if(success) {
        const vector<EpsilonNumber<T>>& distance = certificate;
        size_t j = 0;
        for(size_t i = 0; i < g.num_edges(); i++) {
            while(j < g.num_nodes() && g.offset[j + 1] == i) {
                j++;
            }
            if(g.value[i] + distance[j] < distance[g.adj[i]]) {
                return false;
            }
        }
        return true;
    } else {
        return certificate.empty();
    }
}

double get_time() {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return double(c.QuadPart) / double(f.QuadPart);
#else
    struct timespec c;
    clock_gettime(CLOCK_MONOTONIC, &c);
    return double(c.tv_sec) + double(c.tv_nsec) * 1e-9;
#endif
}

int main(int argc, char* argv[]) {
#define print_line(CATEGORY, TIME, LAST)                                                                                           \
    cout << "    \"" << CATEGORY << "\": " << fixed << setprecision(15) << (TIME) << (LAST ? "\n" : ",\n")
    using namespace std;
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    double start, end;
    uint64_t mask = (argc == 2) ? (atoi(argv[1])) : 63;
    unordered_map<string, uint32_t> mapper;
    vector<string> inv_mapper;
    vector<tuple<uint32_t, uint32_t, base_type_t, bool>> edges_int;
    string fi, mi, se, si;
    base_type_t va;
    start = get_time();
    while(cin.good()) {
        cin >> fi >> mi >> se >> si >> va;
        assert(mi == "-");
        if(!mapper.count(fi)) {
            mapper[fi] = inv_mapper.size();
            inv_mapper.push_back(fi);
        }
        if(!mapper.count(se)) {
            mapper[se] = inv_mapper.size();
            inv_mapper.push_back(se);
        }
        if(si == "<=") {
            edges_int.emplace_back(mapper[se], mapper[fi], va, false);
        } else if(si == "<") {
            edges_int.emplace_back(mapper[se], mapper[fi], va, true);
        } else if(si == ">=") {
            edges_int.emplace_back(mapper[fi], mapper[se], -va, false);
        } else if(si == ">") {
            edges_int.emplace_back(mapper[fi], mapper[se], -va, true);
        } else if(si == "=" || si == "==") {
            edges_int.emplace_back(mapper[se], mapper[fi], va, false);
            edges_int.emplace_back(mapper[fi], mapper[se], -va, false);
        } else {
            assert(!"Wrong sign");
        }
    }
    uint32_t n = inv_mapper.size();
    cerr << "READ_TIME: " << get_time() - start << endl;

    cerr << "NODES: " << n << endl;
    cerr << "EDGES: " << edges_int.size() << endl;
    vector<tuple<uint32_t, uint32_t, EpsilonNumber<base_type_t>, bool>> edges_eps;
    edges_eps.reserve(edges_int.size());
    for(size_t i = 0; i < edges_int.size(); i++) {
        edges_eps.emplace_back(get<0>(edges_int[i]), get<1>(edges_int[i]), EpsilonNumber<base_type_t>(get<2>(edges_int[i])),
                               get<3>(edges_int[i]));
    }
    Graph<base_type_t> g_int(n, edges_int);
    Graph<EpsilonNumber<base_type_t>> g_eps(n, edges_eps);
    canonize(g_eps);
    vector<pair<string, double>> result;
    optional<bool> ok = nullopt;
    optional<vector<EpsilonNumber<base_type_t>>> distance_eps = nullopt;
    optional<vector<base_type_t>> distance_int = nullopt;

    cout << "{" << endl;

    if((mask >> 0) & 1) {
        distance_eps = nullopt;
        start = get_time();
        distance_eps = BFTO(g_eps);
        end = get_time();
        if(!ok) {
            *ok = bool(distance_eps);
        }
        assert(*ok == bool(distance_eps));
        assert(validate(g_eps, distance_eps ? *distance_eps : vector<EpsilonNumber<base_type_t>>(), *ok));
        print_line("EPS_BFTO", end - start, !(mask >> 1));
    }

    if((mask >> 1) & 1) {
        distance_eps = nullopt;
        start = get_time();
        distance_eps = GORC(g_eps);
        end = get_time();
        if(!ok) {
            *ok = bool(distance_eps);
        }
        assert(*ok == bool(distance_eps));
        assert(validate(g_eps, distance_eps ? *distance_eps : vector<EpsilonNumber<base_type_t>>(), *ok));
        print_line("EPS_GORC", end - start, !(mask >> 2));
    }

    if((mask >> 2) & 1) {
        distance_eps = nullopt;
        start = get_time();
        distance_eps = BFCT(g_eps);
        end = get_time();
        if(!ok) {
            *ok = bool(distance_eps);
        }
        assert(*ok == bool(distance_eps));
        assert(validate(g_eps, distance_eps ? *distance_eps : vector<EpsilonNumber<base_type_t>>(), *ok));
        print_line("EPS_BFCT", end - start, !(mask >> 3));
    }

    if((mask >> 3) & 1) {
        distance_eps = nullopt;
        distance_int = nullopt;
        start = get_time();
        distance_int = BFTO(g_int);
        if(distance_int) {
            distance_eps = ORRZ(g_int, *distance_int);
        }
        end = get_time();
        if(!ok) {
            *ok = bool(distance_eps) && bool(distance_int);
        }
        assert(*ok == bool(distance_eps) && bool(distance_int));
        assert(validate(g_eps, distance_eps ? *distance_eps : vector<EpsilonNumber<base_type_t>>(), *ok));
        print_line("OUR_BFTO", end - start, !(mask >> 4));
    }

    if((mask >> 4) & 1) {
        distance_eps = nullopt;
        distance_int = nullopt;
        start = get_time();
        distance_int = GORC(g_int);
        if(distance_int) {
            distance_eps = ORRZ(g_int, *distance_int);
        }
        end = get_time();
        if(!ok) {
            *ok = bool(distance_eps) && bool(distance_int);
        }
        assert(*ok == bool(distance_eps) && bool(distance_int));
        assert(validate(g_eps, distance_eps ? *distance_eps : vector<EpsilonNumber<base_type_t>>(), *ok));
        print_line("OUR_GORC", end - start, !(mask >> 5));
    }

    if((mask >> 5) & 1) {
        distance_eps = nullopt;
        distance_int = nullopt;
        start = get_time();
        distance_int = BFCT(g_int);
        if(distance_int) {
            distance_eps = ORRZ(g_int, *distance_int);
        }
        end = get_time();
        if(!ok) {
            *ok = bool(distance_eps) && bool(distance_int);
        }
        assert(*ok == bool(distance_eps) && bool(distance_int));
        assert(validate(g_eps, distance_eps ? *distance_eps : vector<EpsilonNumber<base_type_t>>(), *ok));
        print_line("OUR_BFCT", end - start, !(mask >> 6));
    }

    cout << "}" << endl;

    cerr << "VERDICT: " << boolalpha << *ok << endl;
    return 0;
#undef print_line
}
